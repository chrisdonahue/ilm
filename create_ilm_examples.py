from collections import Counter
import os
import random

from ilm.mask.util import masked_spans_bounds_valid, masked_spans_overlap


def randomly_mask_document(
    doc,
    masker,
    num_examples,
    max_num_retries,
    min_masked_spans=None,
    max_masked_spans=None,
    random_sample_down_to_max=True,
    ensure_valid_bounds_in_spans=True,
    ensure_nonoverlapping_spans=True,
    ensure_unique=True):
  error_to_count = Counter()
  doc_masks = []
  doc_masks_set = set()

  def mask_acceptable(masked_spans):
    if min_masked_spans is not None and len(masked_spans) < min_masked_spans:
      return False, 'Too few spans'

    if max_masked_spans is not None and len(masked_spans) > max_masked_spans:
      return False, 'Too many spans'

    if ensure_valid_bounds_in_spans and not masked_spans_bounds_valid(masked_spans, len(doc)):
      return False, 'Masked span boundaries are invalid'

    if ensure_nonoverlapping_spans and masked_spans_overlap(masked_spans):
      return False, 'Masked spans overlap'

    if ensure_unique and masked_spans in doc_masks_set:
      return False, 'Mask is not unique'

    return True, None

  for i in range(num_examples):
    mask = None
    num_retries = 0
    while num_retries < max_num_retries and mask is None:
      try:
        mask = tuple(masker.mask(doc))
      except Exception as e:
        error_to_count['Mask function exception: {}'.format(str(e))] += 1
        mask = None

      if mask is not None:
        if max_masked_spans is not None and random_sample_down_to_max and len(mask) > max_masked_spans:
          mask = tuple(random.sample(mask, max_masked_spans))
        mask_is_acceptable, error_msg = mask_acceptable(mask)
        if not mask_is_acceptable:
          error_to_count['Issue with example: {}'.format(error_msg)] += 1
          mask = None

      num_retries += 1

    if mask is not None:
      doc_masks.append(mask)
      doc_masks_set.add(mask)

  return [list(m) for m in doc_masks], error_to_count


def randomly_mask_dataset(
    docs,
    masker,
    num_examples_per_document,
    max_num_retries,
    tqdm=lambda x: x,
    **kwargs):
  docs_masked = []

  error_to_count_total = Counter()

  num_retries_total = 0
  for doc in tqdm(docs):
    doc_masks, error_to_count = randomly_mask_document(
        doc,
        masker,
        num_examples_per_document,
        max_num_retries,
        **kwargs)
    docs_masked.append((doc, doc_masks))
    for k, v in error_to_count.items():
      error_to_count_total[k] += v

  return docs_masked, error_to_count_total


if __name__ == '__main__':
  from argparse import ArgumentParser
  import importlib
  import pickle
  import sys

  from tqdm import tqdm
  
  from ilm.datasets import Dataset, get_dataset
  import ilm.mask
  from ilm.mask.util import mask_cls_str_to_type

  parser = ArgumentParser()

  parser.add_argument('tag', type=str)
  parser.add_argument('out_dir', type=str)
  parser.add_argument('--seed', type=int)

  data_args = parser.add_argument_group('Dataset')
  data_args.add_argument('--data_name', type=str, choices=[t.name.lower() for t in Dataset])
  data_args.add_argument('--data_dir', type=str)
  data_args.add_argument('--data_split', type=str)

  mask_args = parser.add_argument_group('Mask')
  mask_args.add_argument('--mask_cls', type=str)
  mask_args.add_argument('--mask_arg0', type=float)

  parser.add_argument('--max_num_documents', type=int)
  parser.add_argument('--num_examples_per_document', type=int)
  parser.add_argument('--max_num_retries_per_example', type=int)
  parser.add_argument('--min_masked_spans_per_example', type=int)
  parser.add_argument('--max_masked_spans_per_example', type=int)
  parser.add_argument('--allow_duplicate_examples', action='store_false', dest='ensure_unique_examples')

  parser.set_defaults(
      seed=None,
      data_name='arxiv_cs_abstracts',
      data_dir=None,
      data_split='train',
      mask_cls='ilm.mask.hierarchical.MaskHierarchical',
      mask_arg0=None,
      max_num_documents=None,
      num_examples_per_document=16,
      max_num_retries_per_example=16,
      min_masked_spans_per_example=None,
      max_masked_spans_per_example=None,
      ensure_unique_examples=True)
  
  args = parser.parse_args()

  # Set seed
  seed = args.seed
  if seed is None:
    seed = random.randint(0, 1e6)
    print('Random seed {}'.format(seed))
  random.seed(seed)

  # Load data
  dataset = Dataset[args.data_name.upper()]
  docs = get_dataset(
      dataset,
      args.data_split,
      data_dir=args.data_dir,
      shuffle=True,
      limit=args.max_num_documents)

  # Create mask function
  mask_type = mask_cls_str_to_type(args.mask_cls)
  if args.mask_arg0 is None:
    masker = mask_type()
  else:
    masker = mask_type(args.mask_arg0)

  # Create examples
  masked_data, error_to_count = randomly_mask_dataset(
      docs,
      masker,
      args.num_examples_per_document,
      max_num_retries=args.max_num_retries_per_example,
      min_masked_spans=args.min_masked_spans_per_example,
      max_masked_spans=args.max_masked_spans_per_example,
      random_sample_down_to_max=True,
      ensure_valid_bounds_in_spans=True,
      ensure_nonoverlapping_spans=True,
      ensure_unique=args.ensure_unique_examples,
      tqdm=tqdm)

  # Print stats
  num_documents = len(docs)
  num_masked_examples = sum([len(exs) for d, exs in masked_data])
  num_masked_examples_expected = len(docs) * args.num_examples_per_document
  print('Processed {} documents and created {} examples per document (expected {})'.format(
    num_documents,
    num_masked_examples / num_documents,
    args.num_examples_per_document))
  num_retries = sum(error_to_count.values())
  if num_retries > 0:
    print('Errors which caused retries:')
    for k, v in error_to_count.items():
      print('* ({} retries) {}'.format(v, k))
  num_chars_total = 0
  num_chars_masked = 0
  for doc, char_masks in masked_data:
    num_chars_total += len(doc) * len(char_masks)
    for masked_spans in char_masks:
      num_chars_masked += sum([l for _, _, l in masked_spans])
  print('Mask rate (characters): {:.4f}'.format(num_chars_masked / num_chars_total))

  # Save examples
  if not os.path.isdir(args.out_dir):
    os.makedirs(args.out_dir)
  with open(os.path.join(args.out_dir, '{}.pkl'.format(args.tag)), 'wb') as f:
    pickle.dump(masked_data, f)
