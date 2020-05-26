from enum import Enum
from collections import defaultdict
import multiprocessing
import os
import pickle
import random
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel, AdamW, CONFIG_NAME, WEIGHTS_NAME
try:
  import wandb
except:
  pass

import ilm.constants
import ilm.mask
import ilm.mask.util
import ilm.tokenize_util

class Task(Enum):
  # Example: She ate <?> for <?><S>cereal<E>breakfast<E>
  ILM = 0
  # Example: <S>cereal<E>breakfast<E>
  NO_CONTEXT_ILM = 1
  # Example: She ate <?> for <?><S>She ate cereal for breakfast<E>
  NAIVE = 2
  # Example: <S>She ate cereal for breakfast<E>
  LM = 3
  # Example: <S>breakfast for cereal ate She<E>
  REVERSE_LM = 4
  # TODO: NAIVE with no stopwords?


class TargetType(Enum):
  PAD = 0
  CONTEXT = 1
  CONTEXT_SPECIAL = 2
  CONTEXT_INFILL_SEP = 3
  INFILL = 4
  INFILL_SPECIAL = 5
  INFILL_REDUNDANT = 6


def set_random_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)


# NOTE: Multiprocessing pickle/closure issue workaround
_GLOBAL_WORKER_TARGET = None
def _worker_target(doc):
  return _GLOBAL_WORKER_TARGET(doc)
def worker_target_factory(
    tokenizer,
    start_infill_id,
    end_infill_id,
    mask_type_to_id,
    sequence_length,
    task,
    skip_naive_incomplete):
  def fn(doc_and_char_masks):
    doc, char_masks = doc_and_char_masks
    try:
      return doc_and_char_masks_to_input_and_tt(
          doc,
          char_masks,
          tokenizer,
          start_infill_id,
          end_infill_id,
          mask_type_to_id,
          task,
          sequence_length,
          skip_naive_incomplete)
    except Exception as e:
      print(e)
      return None
  return fn


def doc_and_char_masks_to_input_and_tt(
    doc,
    char_masks,
    tokenizer,
    start_infill_id,
    end_infill_id,
    mask_type_to_id,
    task,
    sequence_length,
    skip_naive_incomplete):
  # Tokenize document
  try:
    doc_tokens = ilm.tokenize_util.tokenize(doc, tokenizer=tokenizer)
    doc_tokens_ids = ilm.tokenize_util.tokens_to_ids(doc_tokens, tokenizer=tokenizer)
  except:
    doc_tokens = None
    #error_to_count['Failed to tokenize document'] += len(char_masks)

  # Align character masks to tokens
  tok_masks = []
  if doc_tokens is not None:
    for char_mask in char_masks:
      try:
        tok_mask = ilm.mask.util.align_char_mask_to_tokens(doc, doc_tokens, char_mask)
      except:
        #error_to_count['Failed to align character-level mask to tokens'] += 1
        continue
      tok_masks.append(tok_mask)

  # Apply masks
  contexts_and_answers = []
  for tok_mask in tok_masks:
    try:
      ca = ilm.mask.util.apply_masked_spans(
          doc_tokens_ids,
          tok_mask,
          mask_type_to_id)
    except:
      #error_to_count['Failed to apply mask'] += 1
      continue
    contexts_and_answers.append((tok_mask, ca))

  # Skip examples that would be incomplete for Task.NAIVE (typically the longest task)
  if skip_naive_incomplete:
    contexts_and_answers = [(m, (c, a)) for m, (c, a) in contexts_and_answers if (len(c) + 1 + len(doc_tokens_ids) + 1) <= sequence_length]

  special_ids = set([start_infill_id, end_infill_id] + list(mask_type_to_id.values()))

  inputs = np.zeros((len(contexts_and_answers), sequence_length), dtype=np.uint16)
  tts = np.full((len(contexts_and_answers), sequence_length), TargetType.PAD.value, dtype=np.uint8)
  for i, (mask, (context, answers)) in enumerate(contexts_and_answers):
    # Create example
    example = []

    # (Masked) Context
    if task in [Task.ILM, Task.NAIVE]:
      # Example: She ate <?> for <?>
      example += context

    # Context / answer separator
    context_len = len(example)
    # Example: <S>
    example += [start_infill_id]

    # Answers
    if task in [Task.ILM, Task.NO_CONTEXT_ILM]:
      # Example: cereal<E>breakfast<E>
      for mask_type, answer in answers:
        example += answer
        example += [end_infill_id]
    elif task in [Task.NAIVE, Task.LM]:
      # Example: She ate cereal for breakfast<E>
      example += doc_tokens_ids
      example += [end_infill_id]
    elif task == Task.REVERSE_LM:
      # Example: breakfast for cereal ate She<E>
      example += doc_tokens_ids[::-1]
      example += [end_infill_id]
    else:
      assert False

    if len(example) > sequence_length:
      example = example[:sequence_length]
      #warning_to_count['Example longer than sequence length'] += 1

    # Find special tokens
    context_special_idxs = [l for l, t in enumerate(example) if l < context_len and t in special_ids]
    infill_special_idxs = [l for l, t in enumerate(example) if l > context_len and t in special_ids]

    # Store example in output array
    if len(example) > 0 and (min(example) < np.iinfo(inputs.dtype).min or max(example) > np.iinfo(inputs.dtype).max):
      raise ValueError('Example cannot be stored in numpy array')
    inputs[i, :len(example)] = example

    # Store target types in output array
    tts[i, :context_len] = TargetType.CONTEXT.value
    for l in context_special_idxs:
      tts[i, l] = TargetType.CONTEXT_SPECIAL.value
    tts[i, context_len:context_len+1] = TargetType.CONTEXT_INFILL_SEP.value
    if task in [Task.NAIVE, Task.LM, Task.REVERSE_LM]:
      tts[i, context_len+1:len(example)] = TargetType.INFILL_REDUNDANT.value
      if task == Task.REVERSE_LM:
        mask = mask[::-1]
      for (_, tok_off, tok_len) in mask:
        if task == Task.REVERSE_LM:
          tok_off = (len(doc_tokens_ids) - 1) - (tok_off + tok_len - 1)
        tts[i, context_len+1+tok_off:context_len+1+tok_off+tok_len] = TargetType.INFILL.value
        tts[i, context_len+1+tok_off+tok_len:context_len+1+tok_off+tok_len+1] = TargetType.INFILL_SPECIAL.value
    else:
      tts[i, context_len+1:len(example)] = TargetType.INFILL.value
      for l in infill_special_idxs:
        tts[i, l] = TargetType.INFILL_SPECIAL.value

  return inputs, tts


def masked_dataset_to_inputs_and_tts(
    split,
    tokenizer,
    start_infill_id,
    end_infill_id,
    mask_type_to_id,
    args):
  assert split in ['train', 'eval']
  if split == 'train':
    examples_tag = args.train_examples_tag
    sequence_length = args.train_sequence_length
    max_num_examples = args.train_max_num_examples
    skip_naive_incomplete = args.train_skip_naive_incomplete
  else:
    examples_tag = args.eval_examples_tag
    sequence_length = args.eval_sequence_length
    max_num_examples = args.eval_max_num_examples
    skip_naive_incomplete = args.eval_skip_naive_incomplete

  with open(os.path.join(args.examples_dir, '{}.pkl'.format(examples_tag)), 'rb') as f:
    dataset = pickle.load(f)
  num_docs = len(dataset)

  # Mask and tokenize documents
  global _GLOBAL_WORKER_TARGET
  _GLOBAL_WORKER_TARGET = worker_target_factory(
      tokenizer,
      start_infill_id,
      end_infill_id,
      mask_type_to_id,
      sequence_length,
      Task[args.task.upper()],
      skip_naive_incomplete)
  with multiprocessing.Pool(args.data_loader_num_workers) as p:
    docs_inputs_and_tts = list(tqdm(
      p.imap(_worker_target, dataset),
      total=len(dataset)))

  inputs = np.concatenate([i for i, _ in docs_inputs_and_tts], axis=0)
  tts = np.concatenate([t for _, t in docs_inputs_and_tts], axis=0)

  # TODO: Don't bother doing all the work if we're not going to use it
  if max_num_examples is not None:
    set_random_seed(args.seed)
    example_ids = random.sample(list(range(inputs.shape[0])), max_num_examples)
    inputs = np.take(inputs, example_ids, axis=0)
    tts = np.take(tts, example_ids, axis=0)

  return inputs, tts, num_docs


def tts_to_labels(inputs, tts, label_tts):
  selector = torch.zeros_like(inputs, dtype=torch.bool)
  for tt in label_tts:
    selector |= tts == tt.value
  return torch.where(
      selector,
      inputs,
      torch.full_like(inputs, -1))


def train(args):
  # Init device
  n_gpu = torch.cuda.device_count()
  if n_gpu == 0:
    warnings.warn('No GPU detected. Training on CPU will be very slow')
  elif n_gpu > 1:
    warnings.warn('This codebase is not optimized for multi GPU usage')
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Lambda for filenames
  example_tag_to_fp = lambda tag: os.path.join(args.examples_dir, '{}.pkl'.format(tag))
  out_fn_to_fp = lambda fn: os.path.join(args.train_dir, fn)

  # Create training dir
  os.makedirs(args.train_dir, exist_ok=True)
  resuming = os.path.exists(out_fn_to_fp('step.pkl'))

  # Create tokenizer
  tokenizer = ilm.tokenize_util.Tokenizer[args.tokenizer_name.upper()]
  if tokenizer == ilm.tokenize_util.Tokenizer.CUSTOM:
    ilm.tokenize_util.set_custom_vocab_fp(args.tokenizer_custom_vocab_fp)

  # Update tokenizer
  base_vocab_size = ilm.tokenize_util.vocab_size(tokenizer)
  start_infill_id = base_vocab_size + 0
  end_infill_id = base_vocab_size + 1
  additional_ids_to_tokens = {
      start_infill_id: '<|startofinfill|>',
      end_infill_id: '<|endofinfill|>'
  }
  mask_cls = ilm.mask.util.mask_cls_str_to_type(args.mask_cls)
  mask_types = mask_cls.mask_types()
  mask_type_to_id = {}
  for i, t in enumerate(mask_types):
    t_id = base_vocab_size + 2 + i
    t_tok = '<|infill_{}|>'.format(mask_cls.mask_type_serialize(t))
    additional_ids_to_tokens[t_id] = t_tok
    mask_type_to_id[t] = t_id
  print(additional_ids_to_tokens)
  vocab_size = ilm.tokenize_util.update_tokenizer(additional_ids_to_tokens, tokenizer)
  with open(out_fn_to_fp('additional_ids_to_tokens.pkl'), 'wb') as f:
    pickle.dump(additional_ids_to_tokens, f)

  # Load training data
  if not args.eval_only:
    print('Loading training data')
    loaded_from_cache = False
    if args.data_cache:
      try:
        train_inputs = np.load(out_fn_to_fp('train_inp.npy'))
        train_tts = np.load(out_fn_to_fp('train_tts.npy'))
        with open(out_fn_to_fp('train_num_docs.pkl'), 'rb') as f:
          train_num_docs = pickle.load(f)
        loaded_from_cache = True
      except:
        pass
    if not loaded_from_cache:
      train_inputs, train_tts, train_num_docs = masked_dataset_to_inputs_and_tts(
          'train',
          tokenizer,
          start_infill_id,
          end_infill_id,
          mask_type_to_id,
          args)
      if args.data_cache:
        np.save(out_fn_to_fp('train_inp.npy'), train_inputs)
        np.save(out_fn_to_fp('train_tts.npy'), train_tts)
        with open(out_fn_to_fp('train_num_docs.pkl'), 'wb') as f:
          pickle.dump(train_num_docs, f)
    train_tt_to_count = {TargetType(k):v for k, v in zip(*np.unique(train_tts, return_counts=True))}
    print(train_tt_to_count)
    num_unmasked = train_tt_to_count.get(TargetType.CONTEXT, 0)
    num_masked = train_tt_to_count.get(TargetType.INFILL, 0)
    print('Mask rate (tokens): {:.4f}'.format(num_masked / (num_unmasked + num_masked)))
    print('{} documents, {} examples'.format(train_num_docs, train_inputs.shape[0]))
    print(train_inputs.shape, train_inputs.dtype, train_tts.shape, train_tts.dtype)
    train_data = TensorDataset(
        torch.from_numpy(train_inputs.astype(np.int64)),
        torch.from_numpy(train_tts))
    del train_inputs
    del train_tts

  # Load eval data
  print('Loading eval data')
  loaded_from_cache = False
  if args.data_cache:
    try:
      eval_inputs = np.load(out_fn_to_fp('eval_inp.npy'))
      eval_tts = np.load(out_fn_to_fp('eval_tts.npy'))
      with open(out_fn_to_fp('eval_num_docs.pkl'), 'rb') as f:
        eval_num_docs = pickle.load(f)
      loaded_from_cache = True
    except:
      pass
  if not loaded_from_cache:
    eval_inputs, eval_tts, eval_num_docs = masked_dataset_to_inputs_and_tts(
        'eval',
        tokenizer,
        start_infill_id,
        end_infill_id,
        mask_type_to_id,
        args)
    if args.data_cache:
      np.save(out_fn_to_fp('eval_inp.npy'), eval_inputs)
      np.save(out_fn_to_fp('eval_tts.npy'), eval_tts)
      with open(out_fn_to_fp('eval_num_docs.pkl'), 'wb') as f:
        pickle.dump(eval_num_docs, f)
  eval_tt_to_count = {TargetType(k):v for k, v in zip(*np.unique(eval_tts, return_counts=True))}
  print(eval_tt_to_count)
  num_unmasked = eval_tt_to_count.get(TargetType.CONTEXT, 0)
  num_masked = eval_tt_to_count.get(TargetType.INFILL, 0)
  print('Mask rate (tokens): {:.4f}'.format(num_masked / (num_unmasked + num_masked)))
  print('{} documents, {} examples'.format(eval_num_docs, eval_inputs.shape[0]))
  print(eval_inputs.shape, eval_inputs.dtype, eval_tts.shape, eval_tts.dtype)
  eval_data = TensorDataset(
      torch.from_numpy(eval_inputs.astype(np.int64)),
      torch.from_numpy(eval_tts))
  del eval_inputs
  del eval_tts

  # Calculate number of steps to train for (return if we're just pre-cacheing data)
  if args.train_num_epochs is not None:
    train_num_batches = int(float(train_num_docs * args.train_num_epochs) / args.train_batch_size)
    if train_num_batches == 0:
      return
    print('Maximum number of training steps: {}'.format(train_num_batches / args.train_batch_accumulation))

  # Create data iterators
  print('Creating datasets')
  if not args.eval_only:
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size, drop_last=True)
  eval_sampler = SequentialSampler(eval_data)
  eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size, drop_last=True)

  # Load model
  print('Initializing model...')
  set_random_seed(args.seed)
  if args.model_name in ilm.constants.GPT2_MODEL_NAMES:
    model_type = GPT2LMHeadModel
    cfg_type = GPT2Config
  if resuming:
    print('from saved checkpoint (resuming)')
    model = model_type.from_pretrained(args.train_dir)
  else:
    if args.train_from_scratch:
      print('from scratch')
      cfg = cfg_type.from_pretrained(args.model_name)
      model = model_type(cfg)
    else:
      print('from pretrained checkpoint')
      model = model_type.from_pretrained(args.model_name)
  model.resize_token_embeddings(vocab_size)
  model.to(device)
  model.train()

  # Reset random seed in case model init triggered RNG

  # Initialize optimizers
  if not args.eval_only:
    params = list(model.named_parameters())
    no_decay = ['bias', 'ln']
    optimizer_grouped_parameters = [
      {
        'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
        'weight_decay': args.train_weight_decay
      },
      {
        'params': [p for n, p in params if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0
      }
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.train_learning_rate,
        eps=args.train_adam_epsilon)
    if resuming:
      optimizer.load_state_dict(torch.load(out_fn_to_fp('optimizer.pt')))

  # Create global step
  if resuming:
    with open(out_fn_to_fp('step.pkl'), 'rb') as f:
      step = pickle.load(f)
  else:
    step = 0

  if args.eval_only:
    print('Evaluating')
    model.eval()

    eval_start = time.time()
    eval_token_counts = defaultdict(int)
    eval_token_loss_sums = defaultdict(float)
    for i, eval_batch in enumerate(eval_dataloader):
      with torch.no_grad():
        eval_inputs, eval_tts = tuple(t.to(device) for t in eval_batch)
        eval_logits, _ = model(eval_inputs)
        eval_logits_relevant = eval_logits[:, :-1].contiguous().view(-1, eval_logits.shape[-1])

        for tag, tts in [
            ('context', [TargetType.CONTEXT]),
            ('infill', [TargetType.INFILL, TargetType.INFILL_SPECIAL]),
            ('infill_textonly', [TargetType.INFILL])]:
          eval_labels = tts_to_labels(eval_inputs, eval_tts, tts)
          eval_labels_relevant = eval_labels[:, 1:]
          eval_labels_relevant_count = (eval_labels_relevant != -1).long().sum().item()
          eval_labels_loss = F.cross_entropy(
              eval_logits_relevant,
              eval_labels_relevant.contiguous().view(-1),
              ignore_index=-1).item()
          eval_token_counts[tag] += eval_labels_relevant_count
          eval_token_loss_sums[tag] += eval_labels_loss * eval_labels_relevant_count

    eval_dict = {}
    for tag, count in eval_token_counts.items():
      loss = eval_token_loss_sums[tag]
      if count > 0:
        loss /= count
      eval_dict['eval_{}_count'.format(tag)] = count
      eval_dict['eval_{}_loss'.format(tag)] = loss
    eval_dict['eval_time'] = time.time() - eval_start

    print('-' * 80)
    print('(Step {}) Eval'.format(step))
    for k, v in eval_dict.items():
      print('{}: {}'.format(k, v))
    if args.wandb:
      wandb.log(eval_dict, step=step)

  else:
    print('Training')
    set_random_seed(args.seed)
    best_eval_loss = None
    num_save = -1
    num_summary = -1
    num_batches_complete = step * args.train_batch_accumulation
    start = time.time()
    while True:
      if args.train_num_epochs is not None and num_batches_complete >= train_num_batches:
        break

      for batch in train_dataloader:
        if args.train_num_epochs is not None and num_batches_complete >= train_num_batches:
          break

        elapsed = time.time() - start

        # Evaluate
        if int(elapsed / args.train_eval_secs) > num_save:
          num_save = int(elapsed / args.train_eval_secs)

          model.eval()

          eval_start = time.time()
          eval_token_counts = defaultdict(int)
          eval_token_loss_sums = defaultdict(float)
          for i, eval_batch in enumerate(eval_dataloader):
            with torch.no_grad():
              eval_inputs, eval_tts = tuple(t.to(device) for t in eval_batch)
              eval_logits, _ = model(eval_inputs)
              eval_logits_relevant = eval_logits[:, :-1].contiguous().view(-1, eval_logits.shape[-1])

              for tag, tts in [
                  ('context', [TargetType.CONTEXT]),
                  ('infill', [TargetType.INFILL, TargetType.INFILL_SPECIAL]),
                  ('infill_textonly', [TargetType.INFILL])]:
                eval_labels = tts_to_labels(eval_inputs, eval_tts, tts)
                eval_labels_relevant = eval_labels[:, 1:]
                eval_labels_relevant_count = (eval_labels_relevant != -1).long().sum().item()
                eval_labels_loss = F.cross_entropy(
                    eval_logits_relevant,
                    eval_labels_relevant.contiguous().view(-1),
                    ignore_index=-1).item()
                eval_token_counts[tag] += eval_labels_relevant_count
                eval_token_loss_sums[tag] += eval_labels_loss * eval_labels_relevant_count

          eval_dict = {}
          for tag, count in eval_token_counts.items():
            loss = eval_token_loss_sums[tag]
            if count > 0:
              loss /= count
            eval_dict['eval_{}_count'.format(tag)] = count
            eval_dict['eval_{}_loss'.format(tag)] = loss
          eval_dict['eval_time'] = time.time() - eval_start

          print('-' * 80)
          print('(Step {}) Eval'.format(step))
          for k, v in eval_dict.items():
            print('{}: {}'.format(k, v))
          if args.wandb:
            wandb.log(eval_dict, step=step)

          if best_eval_loss is None or eval_dict['eval_infill_loss'] < best_eval_loss:
            print('Saving')
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.config.to_json_file(out_fn_to_fp(CONFIG_NAME))
            torch.save(model_to_save.state_dict(), out_fn_to_fp(WEIGHTS_NAME))
            torch.save(optimizer.state_dict(), out_fn_to_fp('optimizer.pt'))
            with open(out_fn_to_fp('step.pkl'), 'wb') as f:
              pickle.dump(step, f)
            best_eval_loss = eval_dict['eval_infill_loss']

          model.train()

        # Train
        inputs, tts = tuple(t.to(device) for t in batch)
        # TODO: Option to train on CONTEXT_SPECIAL?
        labels_context = tts_to_labels(inputs, tts, [TargetType.CONTEXT])
        # TODO: Option to skip training on INFILL_REDUNDANT?
        # NOTE: This would give Task.NAIVE/Task.LM less supervision overall but put them more in line with the supervision that Task.ILM and Task.NO_CONTEXT_ILM receive
        labels_infill = tts_to_labels(inputs, tts, [TargetType.INFILL, TargetType.INFILL_SPECIAL, TargetType.INFILL_REDUNDANT])
        logits, _ = model(inputs)
        logits_relevant = logits[:, :-1].contiguous().view(-1, logits.shape[-1])
        loss_context = F.cross_entropy(
            logits_relevant,
            labels_context[:, 1:].contiguous().view(-1),
            ignore_index=-1)
        loss_infill = F.cross_entropy(
            logits_relevant,
            labels_infill[:, 1:].contiguous().view(-1),
            ignore_index=-1)

        loss_context_item = loss_context.item()
        loss_infill_item = loss_infill.item()

        loss = loss_infill
        if args.train_context:
          loss += loss_context

        if args.train_batch_accumulation != 1:
          loss /= float(args.train_batch_accumulation)
        loss.backward()

        # Summarize
        if int(elapsed / args.train_summary_secs) > num_summary:
          num_summary = int(elapsed / args.train_summary_secs)

          print('-' * 80)
          print('(Step {}) Summary'.format(step))
          print(loss_context_item)
          print(loss_infill_item)
          with torch.no_grad():
            for t in inputs, labels_context, labels_infill:
              t0 = list(t[0].cpu().numpy())
              print('-' * 40)
              print(t0)
            for t in inputs, labels_context, labels_infill:
              t0 = list(t[0].cpu().numpy())
              print('-' * 40)
              print(ilm.tokenize_util.decode([0 if t == -1 else t for t in t0], tokenizer))

          if args.wandb:
            wandb.log({
              'loss_context': loss_context_item,
              'loss_infill': loss_infill_item,
            }, step=step)

        if ((num_batches_complete + 1) % args.train_batch_accumulation) == 0:
          torch.nn.utils.clip_grad_norm_(model.parameters(), args.train_max_grad_norm)
          optimizer.step()
          optimizer.zero_grad()
          step += 1

        num_batches_complete += 1


if __name__ == '__main__':
  from argparse import ArgumentParser

  parser = ArgumentParser()

  parser.add_argument('experiment_name', type=str)
  parser.add_argument('train_dir', type=str)
  parser.add_argument('examples_dir', type=str)
  parser.add_argument('--seed', type=int)
  parser.add_argument('--wandb', action='store_true', dest='wandb')
  parser.add_argument('--wandb_project_name', type=str)

  mask_args = parser.add_argument_group('Mask')
  mask_args.add_argument('--mask_cls', type=str)

  tokenizer_args = parser.add_argument_group('Tokenizer')
  tokenizer_args.add_argument('--tokenizer_name', type=str, choices=[t.name.lower() for t in ilm.tokenize_util.Tokenizer])
  tokenizer_args.add_argument('--tokenizer_custom_vocab_fp', type=str)

  task_args = parser.add_argument_group('Task')
  task_args.add_argument('--task', type=str, choices=[t.name.lower() for t in Task])

  data_args = parser.add_argument_group('Data')
  data_args.add_argument('--data_no_cache', action='store_false', dest='data_cache')
  data_args.add_argument('--data_loader_num_workers', type=int)

  model_args = parser.add_argument_group('Model')
  model_args.add_argument('--model_name', type=str, choices=ilm.constants.GPT2_MODEL_NAMES)

  train_args = parser.add_argument_group('Train')
  train_args.add_argument('--train_examples_tag', type=str)
  train_args.add_argument('--train_max_num_examples', type=int)
  train_args.add_argument('--train_num_epochs', type=int)
  train_args.add_argument('--train_from_scratch', action='store_true', dest='train_from_scratch')
  train_args.add_argument('--train_batch_size', type=int)
  train_args.add_argument('--train_batch_accumulation', type=int)
  train_args.add_argument('--train_sequence_length', type=int)
  train_args.add_argument('--train_skip_naive_incomplete', action='store_true', dest='train_skip_naive_incomplete')
  train_args.add_argument('--train_eval_secs', type=float)
  train_args.add_argument('--train_summary_secs', type=float)
  train_args.add_argument('--train_minimal_supervision', action='store_false', dest='train_context')
  train_args.add_argument('--train_learning_rate', type=float)
  train_args.add_argument('--train_weight_decay', type=float)
  train_args.add_argument('--train_adam_epsilon', type=float)
  train_args.add_argument('--train_max_grad_norm', type=float)

  eval_args = parser.add_argument_group('Eval')
  eval_args.add_argument('--eval_only', action='store_true', dest='eval_only')
  eval_args.add_argument('--eval_examples_tag', type=str)
  eval_args.add_argument('--eval_max_num_examples', type=int)
  eval_args.add_argument('--eval_batch_size', type=int)
  eval_args.add_argument('--eval_sequence_length', type=int)
  eval_args.add_argument('--eval_skip_naive_incomplete', action='store_true', dest='eval_skip_naive_incomplete')

  parser.set_defaults(
      seed=None,
      wandb=False,
      wandb_project_name='ilm',
      mask_cls='ilm.mask.hierarchical.MaskHierarchical',
      tokenizer_name='gpt2',
      tokenizer_custom_vocab_fp=None,
      task='ilm',
      data_cache=True,
      data_loader_num_workers=4,
      model_name='gpt2',
      train_examples_tag='train',
      train_max_num_examples=None,
      train_num_epochs=None,
      train_from_scratch=False,
      train_batch_size=20,
      train_batch_accumulation=1,
      train_sequence_length=128,
      train_skip_naive_incomplete=False,
      train_eval_secs=360,
      train_summary_secs=360,
      train_context=True,
      train_learning_rate=5e-5,
      train_weight_decay=0.,
      train_adam_epsilon=1e-8,
      train_max_grad_norm=1.,
      eval_only=False,
      eval_examples_tag='valid',
      eval_max_num_examples=None,
      eval_batch_size=20,
      eval_sequence_length=128,
      eval_skip_naive_incomplete=False)
  
  args = parser.parse_args()

  if args.wandb:
    wandb.init(
        project=args.wandb_project_name,
        name=args.experiment_name)
    wandb.config.update(args)

  if args.seed is None:
    args.seed = random.randint(0, 1e6)
  print('Random seed {}'.format(args.seed))

  train(args)
