import os
import random
import time

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from pytorch_transformers import GPT2Tokenizer
from transformers import GPT2Config, GPT2LMHeadModel, AdamW, WEIGHTS_NAME, CONFIG_NAME
import wandb


class MultiLoader(object):
  def __init__(self, loaders):
    self.loaders = loaders

  def __iter__(self):
    num_domains = len(self.loaders)

    for batches in zip(*self.loaders):
      num_items = len(batches[0])
      batches_concat = [torch.cat([batches[i][j] for i in range(num_domains)], dim=0) for j in range(num_items)]
      yield tuple(batches_concat)


def load_and_encode_paragraphs(
    dataset_fp,
    tokenizer,
    sequence_length,
    domain_id=None,
    allow_incomplete=True,
    special_token_to_prepend=None,
    special_token_to_append=None,
    max_num_batches=None):
  # Open
  with open(dataset_fp, 'r') as f:
    raw_text = f.read()

  # Split
  paragraphs = raw_text.strip().split('\n\n\n')
  if len(paragraphs) == 1:
    paragraphs = raw_text.strip().split('\n\n')

  # Encode
  paragraphs = [tokenizer.encode(p) for p in paragraphs]

  # Prepend special token at end of each paragraph
  if special_token_to_prepend is not None:
    paragraphs = [[special_token_to_prepend] + p for p in paragraphs]

  # Append special token at end of each paragraph
  if special_token_to_append is not None:
    paragraphs = [p + [special_token_to_append] for p in paragraphs]

  # Exclude incomplete
  if not allow_incomplete:
    paragraphs = [p for p in paragraphs if len(p) >= sequence_length]

  # Limit number of batches
  if max_num_batches is not None:
    paragraphs = paragraphs[:max_num_batches]

  # Create batches
  if domain_id is not None:
    domains = np.full((len(paragraphs)), domain_id, dtype=np.int64)
  inputs = np.zeros((len(paragraphs), sequence_length), dtype=np.int64)
  labels = np.full((len(paragraphs), sequence_length), -1, dtype=np.int64)
  for i, paragraph in enumerate(paragraphs):
    paragraph = paragraph[:sequence_length]
    inputs[i, :len(paragraph)] = paragraph
    labels[i, :len(paragraph)] = paragraph

  if not allow_incomplete:
    assert -1 not in labels

  if domain_id is None:
    return tuple(torch.tensor(t) for t in (inputs, labels))
  else:
    return tuple(torch.tensor(t) for t in (domains, inputs, labels))


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

  # Parse datasets
  train_dataset_fps = args.train_data_fp.split(',')
  eval_dataset_fps = args.eval_data_fp.split(',')
  num_domains = len(train_dataset_fps)
  train_num_per_domain = args.train_batch_size // num_domains
  eval_num_per_domain = args.eval_batch_size // num_domains
  if len(eval_dataset_fps) != num_domains:
    raise ValueError()
  if args.train_batch_size % num_domains != 0:
    raise ValueError()
  if args.eval_batch_size % num_domains != 0:
    raise ValueError()

  # Load tokenizer
  tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_dir)

  # Add start tokens
  if args.data_start_tokens:
    domain_start_tokens = ['<|startofdomain{}|>'.format(i) for i in range(num_domains)]
    tokens_dict = {
        'additional_special_tokens': domain_start_tokens
    }
    num_added_tokens = tokenizer.add_special_tokens(tokens_dict)
    domain_start_ids = [tokenizer.convert_tokens_to_ids(s) for s in domain_start_tokens]
    print('Added {} tokens: {} {}'.format(num_added_tokens, domain_start_tokens, domain_start_ids))
  else:
    domain_start_ids = [None for i in range(num_domains)]

  # Use end token optionally
  if args.data_end_tokens:
    eos_token = tokenizer.convert_tokens_to_ids('<|endoftext|>')
  else:
    eos_token = None

  # Save vocabulary
  tokenizer.save_vocabulary(args.train_dir)

  # Load data
  train_dataloaders = []
  eval_dataloaders = []
  for train_fp, eval_fp, prepend_token in zip(train_dataset_fps, eval_dataset_fps, domain_start_ids):
    train_dataset = load_and_encode_paragraphs(
        train_fp,
        tokenizer,
        args.train_sequence_length,
        special_token_to_prepend=prepend_token,
        special_token_to_append=eos_token)
    eval_dataset = load_and_encode_paragraphs(
        eval_fp,
        tokenizer,
        args.eval_sequence_length,
        special_token_to_prepend=prepend_token,
        special_token_to_append=eos_token,
        max_num_batches=None if args.eval_max_num_batches is None else args.eval_max_num_batches * eval_num_per_domain)

    # Create data iterators
    train_data = TensorDataset(*train_dataset)
    train_sampler = RandomSampler(train_data)
    train_dataloaders.append(DataLoader(train_data, sampler=train_sampler, batch_size=train_num_per_domain, drop_last=True))
    eval_data = TensorDataset(*eval_dataset)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloaders.append(DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_num_per_domain, drop_last=True))

  train_dataloader = MultiLoader(train_dataloaders)
  eval_dataloader = MultiLoader(eval_dataloaders)

  # Load model
  if args.train_from_scratch:
    print('Training from scratch')
    config = GPT2Config('gpt2_cfgs/{}.json'.format(args.model_cfg))
    model = GPT2LMHeadModel(config)
    model.config.to_json_file(os.path.join(args.train_dir, CONFIG_NAME))
  else:
    model = GPT2LMHeadModel.from_pretrained(model_name_or_dir)
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

  if args.train_num_epochs is not None:
    train_dataset_len = train_dataset[0].shape[0]
    train_num_steps = int((train_dataset_len * args.train_num_epochs) / float(args.train_batch_size))
    print('Training for {} steps'.format(train_num_steps))

  # Prepare optimizer
  summary_last_time = None
  summary_last_step = None
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
        eval_loss = 0.
        eval_num_batches = 0
        for i, eval_batch in enumerate(eval_dataloader):
          with torch.no_grad():
            eval_inputs, eval_labels = tuple(t.to(device) for t in eval_batch)
            eval_loss += model(eval_inputs, labels=eval_labels)[0].item()
            eval_num_batches += 1

        wandb.log({
          'eval_loss': eval_loss / eval_num_batches,
          'eval_time': time.time() - eval_start
        }, step=step)

        if best_eval_loss is None or eval_loss < best_eval_loss:
          model_to_save = model.module if hasattr(model, 'module') else model
          output_model_fp = os.path.join(args.train_dir, WEIGHTS_NAME)
          torch.save(model_to_save.state_dict(), output_model_fp)
          output_config_fp = os.path.join(args.train_dir, CONFIG_NAME)
          model_to_save.config.to_json_file(output_config_fp)
          best_eval_loss = eval_loss

        model.train()

      # Train
      inputs, labels = tuple(t.to(device) for t in batch)

      loss, _, _ = model(inputs, labels=labels)

      if args.train_batch_accumulation != 1:
        loss /= float(args.train_batch_accumulation)
      loss.backward()

      if ((step + 1) % args.train_batch_accumulation) == 0:
        optimizer.step()
        #scheduler.step()
        optimizer.zero_grad()

      step += 1

      # Summarize
      if int(elapsed / args.train_summary_secs) > num_summary:
        num_summary = int(elapsed / args.train_summary_secs)

        if summary_last_step is None:
          num_steps = step
          num_seconds = elapsed
        else:
          num_steps = step - summary_last_step
          num_seconds = time.time() - summary_last_time
        summary_last_step = step
        summary_last_time = time.time()

        wandb.log({
          'loss': loss.item(),
          'steps_per_second': num_steps / num_seconds
        }, step=step)


if __name__ == '__main__':
  from argparse import ArgumentParser

  parser = ArgumentParser()

  parser.add_argument('experiment_name', type=str)
  parser.add_argument('train_dir', type=str)
  parser.add_argument('--seed', type=int)

  parser.add_argument('--model_cfg', type=str, choices=['gpt2', 'gpt2-medium', 'gpt2-large'])

  data_args = parser.add_argument_group('Data')
  data_args.add_argument('--data_start_tokens', action='store_true', dest='data_start_tokens')
  data_args.add_argument('--data_end_tokens', action='store_true', dest='data_end_tokens')

  train_args = parser.add_argument_group('Train')
  train_args.add_argument('--train_data_fp', type=str)
  train_args.add_argument('--train_from_scratch', action='store_true', dest='train_from_scratch')
  train_args.add_argument('--train_batch_size', type=int)
  train_args.add_argument('--train_batch_accumulation', type=int)
  train_args.add_argument('--train_sequence_length', type=int)
  train_args.add_argument('--train_eval_secs', type=float)
  train_args.add_argument('--train_summary_secs', type=float)
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
      seed=None,
      model_cfg='gpt2',
      data_start_tokens=False,
      data_end_tokens=False,
      train_data_fp=None,
      train_from_scratch=False,
      train_batch_size=512,
      train_batch_accumulation=1,
      train_sequence_length=1024,
      train_eval_secs=360,
      train_summary_secs=360,
      train_learning_rate=6.25e-5,
      train_weight_decay=0.01,
      train_adam_epsilon=1e-8,
      train_num_epochs=None,
      eval_data_fp=None,
      eval_batch_size=1,
      eval_sequence_length=1024,
      eval_max_num_batches=None)
  
  args = parser.parse_args()

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
