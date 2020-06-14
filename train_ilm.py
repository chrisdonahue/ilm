import multiprocessing
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel, AdamW, WarmupLinearSchedule, WEIGHTS_NAME, CONFIG_NAME
try:
  import wandb
except:
  pass

from custom_modules import *
from util import *

def _encode_document(x):
  return encode_document_no_special(*x)


def load_and_encode_paragraphs(
    dataset_fp,
    style,
    sequence_length,
    mask_kwargs,
    task_exclude_context=False,
    task_naive_targets=False,
    task_randomize_infill_order=False,
    task_max_kq=None,
    task_max_ka=None,
    num_tasks_per_example=1,
    num_threads=1,
    max_num_items=None):
  print('Processing {}'.format(dataset_fp))
  print('Loading')
  docs_raw = load_raw_documents(dataset_fp, style)

  # Pre-filter for efficiency
  # NOTE: Hack
  if max_num_items is not None:
    docs_raw = docs_raw[:max_num_items * 2]

  # Split docs
  print('Splitting')
  docs = [split_raw_document(d, style) for d in tqdm(docs_raw)]
  del docs_raw

  # Encode docs
  print('Encoding')
  with multiprocessing.Pool(num_threads) as p:
    docs_enc = list(tqdm(p.imap(_encode_document, zip(docs, [style] * len(docs))), total=len(docs)))
  del docs

  # Mask
  print('Masking')
  docs_masked = []
  for d in tqdm(docs_enc):
    for _ in range(num_tasks_per_example):
      mask = random_mask(d, **mask_kwargs)
      if task_max_kq is not None and len(mask) > task_max_kq:
        mask = random.sample(mask, task_max_kq)
      d_masked = apply_mask(d, mask)
      docs_masked.append(d_masked)

  if task_naive_targets:
    print('Flattening')
    docs_flat = []
    docs_answer_offsets = []
    docs_answer_lens = []
    for i, d in tqdm(enumerate(docs_enc)):
      d = flatten_document_enc(d)
      for j in range(num_tasks_per_example):
        dm, da = docs_masked[(i * num_tasks_per_example) + j]
        dm = flatten_document_enc(dm)
        docs_flat.append((dm, d))

        answer_offsets = []
        for k, tok_id in enumerate(dm):
          if tok_id in INFILL_TYPES_IDS:
            answer_offsets.append(k)
        assert len(answer_offsets) == len(da)
        docs_answer_offsets.append(answer_offsets)

        answer_lens = [len(flatten_document_answers([da[i]])) - 1 for i in range(len(da))]
        docs_answer_lens.append(answer_lens)
    del docs_enc
    del docs_masked

    # Filter
    print('Filtering')
    remove = set([i for i, (d, _) in enumerate(docs_flat) if (len(d) + 1) > sequence_length])
    docs_flat = [x for i, x in enumerate(docs_flat) if i not in remove]
    docs_answer_offsets = [x for i, x in enumerate(docs_answer_offsets) if i not in remove]
    docs_answer_lens = [x for i, x in enumerate(docs_answer_lens) if i not in remove]
    # TODO: Filter these? Right now we're leaving them because it results in forcing the model to predict <|endofinfill|> only as a base case
    #docs_flat = [(c, a) for c, a in docs_flat if len(a) > 0]
  else:
    del docs_enc
    # Flatten
    print('Flattening')
    docs_flat = [(flatten_document_enc(d), flatten_document_answers(a)) for d, a in tqdm(docs_masked)]
    del docs_masked

    # Limit number of answers
    if task_max_ka is not None:
      print('Limiting')
      docs_flat = [limit_answers(d, a, task_max_ka, task_randomize_infill_order) for d, a in tqdm(docs_flat)]

    # Filter
    print('Filtering')
    docs_flat = [(d, a) for d, a in docs_flat if (len(d) + 1) <= sequence_length]
    # TODO: Filter these? Right now we're leaving them because it results in forcing the model to predict <|endofinfill|> only as a base case
    #docs_flat = [(c, a) for c, a in docs_flat if len(a) > 0]

  # Limit number of batches
  if max_num_items is not None:
    docs_flat = docs_flat[:max_num_items]

  # Paired
  print('Pairing')
  if task_exclude_context:
    tasks = [
        [ADDITIONAL_TYPES_TO_ID['start_infill']]
        + a
        + [ADDITIONAL_TYPES_TO_ID['end_infill']]
        for c, a in tqdm(docs_flat)]
  else:
    tasks = [
        c
        + [ADDITIONAL_TYPES_TO_ID['start_infill']]
        + a
        + [ADDITIONAL_TYPES_TO_ID['end_infill']]
        for c, a in tqdm(docs_flat)]
  del docs_flat

  # Create batches
  print('Batching')
  special_tokens_set = set(INFILL_TYPES_TO_ID.values())
  inputs = np.zeros((len(tasks), sequence_length), dtype=np.int64)
  labels_lm = np.full((len(tasks), sequence_length), -1, dtype=np.int64)
  labels_ilm = np.full((len(tasks), sequence_length), -1, dtype=np.int64)
  for i, task in tqdm(enumerate(tasks), total=len(tasks)):
    task = task[:sequence_length]
    infill_start = task.index(ADDITIONAL_TYPES_TO_ID['start_infill'])

    inputs[i, :len(task)] = task

    for j in range(infill_start + 1):
      if task[j] not in special_tokens_set:
        labels_lm[i, j] = task[j]

    labels_ilm[i, :len(task)] = task
    labels_ilm[i, :infill_start+1] = -1

    if task_naive_targets:
      mask = np.zeros_like(labels_ilm[i])
      answer_offsets = docs_answer_offsets[i]
      answer_lens = docs_answer_lens[i]
      running_offset = infill_start + 1
      for ao, al in zip(answer_offsets, answer_lens):
        mask[running_offset+ao:running_offset+ao+al] = 1
        running_offset += al - 1
      labels_ilm[i, mask == 0] = -1

  return tuple(torch.tensor(t) for t in (inputs, labels_lm, labels_ilm))


def train(args):
  # Init device(s)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  n_gpu = torch.cuda.device_count()

  # Create training dir
  if not os.path.isdir(args.train_dir):
    os.makedirs(args.train_dir)
    model_name_or_dir = args.model_cfg
  else:
    print('Resuming')
    model_name_or_dir = args.train_dir

  # Save vocabulary
  tokenizer = get_tokenizer()
  tokenizer.save_vocabulary(args.train_dir)
  if args.task_randomize_infill_order and args.task_max_ka is not None:
    tokenizer_set_max_order_num(args.task_max_ka)

  # Load data
  mask_kwargs = {k:v for k,v in vars(args).items() if k.startswith('mask')}
  train_dataset = load_and_encode_paragraphs(
      args.train_data_fp,
      args.data_style,
      args.train_sequence_length,
      mask_kwargs,
      task_exclude_context=args.task_exclude_context,
      task_naive_targets=args.task_naive_targets,
      task_randomize_infill_order=args.task_randomize_infill_order,
      task_max_kq=args.task_max_kq,
      task_max_ka=args.task_max_ka,
      num_tasks_per_example=args.train_num_tasks_per_example,
      num_threads=args.data_num_threads)
  max_num_items = None
  if args.eval_max_num_batches is not None:
    max_num_items = args.eval_max_num_batches * args.eval_batch_size
  eval_dataset = load_and_encode_paragraphs(
      args.eval_data_fp,
      args.data_style,
      args.eval_sequence_length,
      mask_kwargs,
      task_exclude_context=args.task_exclude_context,
      task_naive_targets=args.task_naive_targets,
      task_randomize_infill_order=args.task_randomize_infill_order,
      task_max_kq=1,
      task_max_ka=1,
      num_tasks_per_example=1,
      num_threads=args.data_num_threads,
      max_num_items=max_num_items)

  # Create data iterators
  train_data = TensorDataset(*train_dataset)
  train_sampler = RandomSampler(train_data)
  train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size, drop_last=True)
  eval_data = TensorDataset(*eval_dataset)
  eval_sampler = SequentialSampler(eval_data)
  eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size, drop_last=True)

  # Load model
  if args.model_phonetics:
    if args.model_tie:
      if args.model_phonetics_known_only:
        model_type = GPT2LMHeadPhoneticsKnownModel
      else:
        model_type = GPT2LMHeadPhoneticsModel
    else:
      if args.model_phonetics_known_only:
        model_type = GPT2LMHeadPhoneticsUntiedKnownModel
      else:
        model_type = GPT2LMHeadPhoneticsUntiedModel
  else:
    if args.model_tie:
      model_type = GPT2LMHeadModel
    else:
      model_type = GPT2LMHeadUntiedModel
  if args.train_from_scratch:
    print('Training from scratch')
    config = GPT2Config('gpt2_cfgs/{}.json'.format(args.model_cfg))
    model = model_type(config)
    model.config.to_json_file(os.path.join(args.train_dir, CONFIG_NAME))
  else:
    model = model_type.from_pretrained(model_name_or_dir)
  model.resize_token_embeddings(len(tokenizer))
  model.to(device)
  model.train()

  # Initialize optimizers
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
  """
  scheduler = WarmupLinearSchedule(
      opt,
      warmup_steps=args.warmup_steps,
      t_total=args.total_steps)
  """

  # Calculate number of steps to train for
  if args.train_num_epochs is not None:
    train_dataset_len = train_dataset[0].shape[0] / float(args.train_num_tasks_per_example)
    train_num_steps = int(float(train_dataset_len * args.train_num_epochs) / args.train_batch_size)
    print('Training for {} steps'.format(train_num_steps))

  # Prepare optimizer
  step = 0
  best_eval_loss = None
  num_save = -1
  num_summary = -1
  start = time.time()
  while True:
    if args.train_num_epochs is not None and step >= train_num_steps:
      break

    for batch in train_dataloader:
      elapsed = time.time() - start

      # Evaluate
      if int(elapsed / args.train_eval_secs) > num_save:
        num_save = int(elapsed / args.train_eval_secs)

        model.eval()

        eval_start = time.time()
        eval_loss_lm = 0.
        eval_loss_ilm = 0.
        eval_num_batches = 0
        for i, eval_batch in enumerate(eval_dataloader):
          with torch.no_grad():
            eval_inputs, eval_labels_lm, eval_labels_ilm = tuple(t.to(device) for t in eval_batch)
            eval_logits, _ = model(eval_inputs)
            eval_logits_relevant = eval_logits[:, :-1].contiguous().view(-1, eval_logits.shape[-1])
            eval_loss_lm_batch = F.cross_entropy(
                eval_logits_relevant,
                eval_labels_lm[:, 1:].contiguous().view(-1),
                ignore_index=-1)
            eval_loss_ilm_batch = F.cross_entropy(
                eval_logits_relevant,
                eval_labels_ilm[:, 1:].contiguous().view(-1),
                ignore_index=-1)
            eval_loss_lm += eval_loss_lm_batch.item()
            eval_loss_ilm += eval_loss_ilm_batch.item()
            eval_num_batches += 1

        print('-' * 80)
        print('(Step {}) Eval'.format(step))
        print(eval_loss_lm / eval_num_batches)
        print(eval_loss_ilm / eval_num_batches)

        if args.wandb:
          wandb.log({
            'eval_loss_lm': eval_loss_lm / eval_num_batches,
            'eval_loss_ilm': eval_loss_ilm / eval_num_batches,
            'eval_time': time.time() - eval_start
          }, step=step)

        if best_eval_loss is None or eval_loss_ilm < best_eval_loss:
          model_to_save = model.module if hasattr(model, 'module') else model
          output_model_fp = os.path.join(args.train_dir, WEIGHTS_NAME)
          torch.save(model_to_save.state_dict(), output_model_fp)
          output_config_fp = os.path.join(args.train_dir, CONFIG_NAME)
          model_to_save.config.to_json_file(output_config_fp)
          best_eval_loss = eval_loss_ilm

        model.train()

      # Train
      inputs, labels_lm, labels_ilm = tuple(t.to(device) for t in batch)

      logits, _ = model(inputs)
      logits_relevant = logits[:, :-1].contiguous().view(-1, logits.shape[-1])
      loss_lm = F.cross_entropy(
          logits_relevant,
          labels_lm[:, 1:].contiguous().view(-1),
          ignore_index=-1)
      loss_ilm = F.cross_entropy(
          logits_relevant,
          labels_ilm[:, 1:].contiguous().view(-1),
          ignore_index=-1)

      loss = loss_ilm
      if args.train_lm:
        loss += loss_lm

      if args.train_batch_accumulation != 1:
        loss /= float(args.train_batch_accumulation)
      loss.backward()

      if ((step + 1) % args.train_batch_accumulation) == 0:
        optimizer.step()
        #scheduler.step()
        optimizer.zero_grad()

      # Summarize
      if int(elapsed / args.train_summary_secs) > num_summary:
        num_summary = int(elapsed / args.train_summary_secs)

        print('-' * 80)
        print('(Step {}) Summary'.format(step))
        print(loss_lm.item())
        print(loss_ilm.item())
        print('-' * 40)
        print(list(inputs[0].cpu().numpy()))
        print('-' * 40)
        print(list(labels_lm[0].cpu().numpy()))
        print('-' * 40)
        print(list(labels_ilm[0].cpu().numpy()))
        print('-' * 40)
        print(tokenizer.decode(list(inputs[0].cpu().numpy())))

        if args.wandb:
          wandb.log({
            'loss_lm': loss_lm.item(),
            'loss_ilm': loss_ilm.item(),
          }, step=step)

      step += 1


if __name__ == '__main__':
  from argparse import ArgumentParser

  parser = ArgumentParser()

  parser.add_argument('experiment_name', type=str)
  parser.add_argument('train_dir', type=str)
  parser.add_argument('--wandb', action='store_true', dest='wandb')
  parser.add_argument('--seed', type=int)

  task_args = parser.add_argument_group('Task')
  task_args.add_argument('--task_exclude_context', action='store_true', dest='task_exclude_context')
  task_args.add_argument('--task_naive_targets', action='store_true', dest='task_naive_targets')
  task_args.add_argument('--task_randomize_infill_order', action='store_true', dest='task_randomize_infill_order')
  task_args.add_argument('--task_max_kq', type=int)
  task_args.add_argument('--task_max_ka', type=int)

  data_args = parser.add_argument_group('Data')
  data_args.add_argument('--data_style', type=str, choices=['verse', 'abstract'])
  data_args.add_argument('--data_num_threads', type=int)

  model_args = parser.add_argument_group('Model')
  model_args.add_argument('--model_cfg', type=str, choices=['gpt2', 'gpt2-medium', 'gpt2-large'])
  model_args.add_argument('--model_phonetics', action='store_true', dest='model_phonetics')
  model_args.add_argument('--model_phonetics_known_only', action='store_true', dest='model_phonetics_known_only')
  model_args.add_argument('--model_untie', action='store_false', dest='model_tie')

  mask_args = parser.add_argument_group('Mask')
  mask_args.add_argument('--mask_document_p', type=float)
  mask_args.add_argument('--mask_paragraph_p', type=float)
  mask_args.add_argument('--mask_sentence_p', type=float)
  mask_args.add_argument('--mask_ngram_p', type=float)
  mask_args.add_argument('--mask_word_p', type=float)
  mask_args.add_argument('--mask_firstword_p', type=float)
  mask_args.add_argument('--mask_leadingwords_p', type=float)
  mask_args.add_argument('--mask_lastword_p', type=float)
  mask_args.add_argument('--mask_ngram_max_n', type=int)

  train_args = parser.add_argument_group('Train')
  train_args.add_argument('--train_data_fp', type=str)
  train_args.add_argument('--train_from_scratch', action='store_true', dest='train_from_scratch')
  train_args.add_argument('--train_num_tasks_per_example', type=int)
  train_args.add_argument('--train_batch_size', type=int)
  train_args.add_argument('--train_batch_accumulation', type=int)
  train_args.add_argument('--train_sequence_length', type=int)
  train_args.add_argument('--train_eval_secs', type=float)
  train_args.add_argument('--train_summary_secs', type=float)
  train_args.add_argument('--train_lm', action='store_true', dest='train_lm')
  train_args.add_argument('--train_learning_rate', type=float)
  train_args.add_argument('--train_weight_decay', type=float)
  train_args.add_argument('--train_adam_epsilon', type=float)
  train_args.add_argument('--train_num_epochs', type=float)

  eval_args = parser.add_argument_group('Eval')
  eval_args.add_argument('--eval_data_fp', type=str)
  eval_args.add_argument('--eval_batch_size', type=int)
  eval_args.add_argument('--eval_sequence_length', type=int)
  eval_args.add_argument('--eval_max_num_batches', type=int)

  parser.set_defaults(
      wandb=False,
      seed=None,
      task_exclude_context=False,
      task_naive_targets=False,
      task_randomize_infill_order=False,
      task_max_kq=None,
      task_max_ka=None,
      data_style='verse',
      data_num_threads=1,
      model_cfg='gpt2',
      model_phonetics=False,
      model_phonetics_known_only=False,
      model_tie=True,
      mask_document_p=0.,
      mask_paragraph_p=0.15,
      mask_sentence_p=0.15,
      mask_ngram_p=0.05,
      mask_word_p=0.15,
      mask_firstword_p=0.,
      mask_leadingwords_p=0.05,
      mask_lastword_p=0.05,
      mask_ngram_max_n=8,
      train_data_fp=None,
      train_from_scratch=False,
      train_num_tasks_per_example=2,
      train_batch_size=512,
      train_batch_accumulation=1,
      train_sequence_length=1024,
      train_eval_secs=360,
      train_summary_secs=360,
      train_lm=False,
      train_learning_rate=6.25e-5,
      train_weight_decay=0.01,
      train_adam_epsilon=1e-8,
      train_num_epochs=None,
      eval_data_fp=None,
      eval_batch_size=1,
      eval_sequence_length=1024,
      eval_max_num_batches=None)
  
  args = parser.parse_args()

  if args.wandb:
    wandb.init(
        project='ilm',
        name=args.experiment_name)
    wandb.config.update(args)

  seed = args.seed
  if seed is None:
    seed = random.randint(0, 1e6)
    print('Random seed {}'.format(seed))
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

  train(args)
