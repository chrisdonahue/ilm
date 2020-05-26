import random
import unittest

from ilm.mask.custom import *

class TestCustom(unittest.TestCase):

  def test_mask_punctuation(self):
    doc = "This is a document; it contains words. a!b@c#d4e5"
    masker = MaskPunctuation()
    random.seed(0)
    masked_spans = masker.mask(doc)
    self.assertEqual(masked_spans, [
      (MaskPunctuationType.SENTENCE_TERMINAL, 40, 1),
      (MaskPunctuationType.OTHER, 42, 1)])

  def test_mask_proper_noun(self):
    doc = "Mary went for a hike in Yosemite"
    masker = MaskProperNoun()
    masked_spans = masker.mask(doc)
    self.assertEqual(masked_spans, [
      (MaskProperNounType.PROPER_NOUN, 0, 4),
      (MaskProperNounType.PROPER_NOUN, doc.index('Yosemite'), 8)])



if __name__ == '__main__':
  unittest.main()
