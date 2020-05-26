from io import BytesIO
import hashlib
import random
import pickle
import unittest

import ilm.datasets
import ilm.mask.hierarchical
from create_ilm_examples import *

class TestCreateIlmExamples(unittest.TestCase):

  def test_randomly_mask_dataset(self):
    random.seed(0)

    docs = ilm.datasets.get_dataset(
        ilm.datasets.Dataset.ROC_STORIES,
        'valid',
        shuffle=True,
        limit=100)
    self.assertTrue('Jen was waiting for a package all day' in docs[0])

    masker = ilm.mask.hierarchical.MaskHierarchical(0.05)

    masked_data, error_to_count = randomly_mask_dataset(
        docs,
        masker,
        8,
        max_num_retries=4,
        min_masked_spans=1,
        max_masked_spans=2,
        random_sample_down_to_max=False)

    self.assertEqual(len(masked_data), 100)
    num_examples = sum([len(exs) for d, exs in masked_data])
    self.assertEqual(num_examples, 720)
    for i, (d, exs) in enumerate(masked_data):
      for j, masked_spans in enumerate(exs):
        if i == 0 and j == 0:
          self.assertEqual(masked_spans, [(ilm.mask.hierarchical.MaskHierarchicalType.WORD, 217, 6)])
        self.assertTrue(len(masked_spans) in [1, 2])

    self.assertEqual(len(error_to_count), 3)
    self.assertEqual(error_to_count['Issue with example: Too many spans'], 738)
    self.assertEqual(error_to_count['Issue with example: Too few spans'], 61)
    self.assertEqual(error_to_count['Issue with example: Mask is not unique'], 47)

    m = hashlib.md5()
    with BytesIO() as masked_data_bytes:
      pickle.dump(masked_data, masked_data_bytes)
      masked_data_bytes.seek(0)
      m.update(masked_data_bytes.getvalue())
    self.assertEqual(m.hexdigest(), '16f503ee270a0f6b61297a2eb3e40648')


if __name__ == '__main__':
  unittest.main()
