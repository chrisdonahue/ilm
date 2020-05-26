import random
import unittest

from ilm.mask.hierarchical import *
from ilm.tokenize_util import tokenize, tokens_to_ids, decode

_DOC = """
After Iowa's first-in-the-nation caucuses melted down earlier this week, all eyes are on the New Hampshire primary to give some clarity to the crowded and chaotic Democratic presidential race.
Voters in the New England state will head to the polls on February 11 to pick the Democrat they want to see nominated. The contest is the second stop after the all-important Iowa caucuses, which normally help cut down the field of candidates.
But issues in the Midwestern state have left the race more open-ended than usual. Historically, the two or three candidates who come out on top in Iowa see increased momentum for the rest of the early-voting season. Now, the muddled results have left the primary race in slight disarray.
"""


class TestHierarchical(unittest.TestCase):

  def setUp(self):
    self.mask_fn = MaskHierarchical(p=0.15)
    self.doc = _DOC

  def test_mask_paper(self):
    mask_types = self.mask_fn.mask_types()
    self.assertEqual(len(mask_types), 5)

    random.seed(4)
    masked_spans = self.mask_fn.mask(self.doc)
    self.assertEqual(len(masked_spans), 14)


if __name__ == '__main__':
  unittest.main()
