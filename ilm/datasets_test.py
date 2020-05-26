import unittest

from ilm.paths import RAW_DATA_DIR
from ilm.datasets import *

class TestDatasets(unittest.TestCase):

  def test_custom(self):
    d = get_dataset(
        Dataset.CUSTOM,
        'train_title',
        data_dir=os.path.join(RAW_DATA_DIR, 'roc_stories'),
        limit=100)
    self.assertEqual(len(d), 100)

  def test_arxiv_cs_abstracts(self):
    expected_split_lens = [153385, 41568, 32655]
    for i, split in enumerate(['train', 'valid', 'test']):
      expected_split_len = expected_split_lens[i]
      d = arxiv_cs_abstracts(split=split)
      self.assertEqual(len(d), expected_split_len)

  def test_roc_stories(self):
    expected_split_lens = [97828, 1866, 1867, 100, 100]
    for i, split in enumerate(['train', 'valid', 'test', 'test_hand_title']):
      expected_split_len = expected_split_lens[i] 
      for with_titles in [False, True]:
        for exclude_nonstandard in [False, True]:
          d = get_dataset(
              Dataset.ROC_STORIES,
              split,
              with_titles=with_titles,
              exclude_nonstandard=exclude_nonstandard)
          self.assertEqual(len(d), expected_split_len)

  def test_lyrics_stanzas(self):
    expected_split_lens = [1741697, 317411, 279758]
    for i, split in enumerate(['train', 'valid', 'test']):
      expected_split_len = expected_split_lens[i] 
      d = get_dataset(Dataset.LYRICS_STANZAS, split)
      self.assertEqual(len(d), expected_split_len)


if __name__ == '__main__':
  unittest.main()
