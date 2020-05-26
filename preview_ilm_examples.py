if __name__ == '__main__':
  from argparse import ArgumentParser
  import importlib
  import os
  import pickle
  import random

  from ilm.mask.util import apply_masked_spans

  parser = ArgumentParser()

  parser.add_argument('tag', type=str)
  parser.add_argument('--examples_dir', type=str)
  parser.add_argument('--seed', type=int)
  parser.add_argument('--max_num_preview', type=int)

  parser.set_defaults(
      examples_dir=None,
      seed=None,
      max_num_preview=8)

  args = parser.parse_args()

  if args.examples_dir is None:
    fp = args.tag
  else:
    fp = os.path.join(args.examples_dir, '{}.pkl'.format(args.tag))
  with open(fp, 'rb') as f:
    masked_data = pickle.load(f)

  random.seed(args.seed)

  if len(masked_data) > args.max_num_preview:
    masked_data = random.sample(masked_data, args.max_num_preview)

  for doc, examples in masked_data:
    if len(examples) == 0:
      continue
    masked_spans = random.choice(examples)
    mask_span_type_to_str = {t:'<|{}|>'.format(str(t)) for t, _, _ in masked_spans}
    context, answers = apply_masked_spans(
        doc,
        masked_spans,
        mask_span_type_to_str)
    for _ in range(4):
      print('-' * 80)
    print(' ' * 36 + '-' * 8)
    print(' ' * 36 + 'ORIGINAL')
    print(' ' * 36 + '-' * 8)
    print(doc)
    print(' ' * 36 + '-' * 7)
    print(' ' * 36 + 'CONTEXT')
    print(' ' * 36 + '-' * 7)
    print(context)
    print(' ' * 36 + '-' * 7)
    print(' ' * 36 + 'ANSWERS')
    print(' ' * 36 + '-' * 7)
    for i, (span_type, span) in enumerate(answers):
      print(mask_span_type_to_str[span_type])
      print(span)
      if i != len(answers) - 1:
        print('-' * 20)
