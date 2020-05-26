import unittest

from ilm.mask.example import *
from ilm.mask.hierarchical import *
from ilm.mask.util import *

class TestUtil(unittest.TestCase):

  def test_mask_cls_str_to_type(self):
    self.assertEqual(
        mask_cls_str_to_type('ilm.mask.example.MaskPunctuation'),
        MaskPunctuation)
    self.assertEqual(
        mask_cls_str_to_type('ilm.mask.example.MaskProperNoun'),
        MaskProperNoun)
    self.assertEqual(
        mask_cls_str_to_type('ilm.mask.hierarchical.MaskHierarchical'),
        MaskHierarchical)

  def test_masked_spans_bounds_valid(self):
    self.assertFalse(masked_spans_bounds_valid([(None, -1, 4)], 6))
    self.assertFalse(masked_spans_bounds_valid([(None, 4, -1)], 6))
    self.assertFalse(masked_spans_bounds_valid([(None, 4, 0)], 6))
    self.assertTrue(masked_spans_bounds_valid([(None, 4, 1)], 6))
    self.assertTrue(masked_spans_bounds_valid([(None, 4, 2)], 6))
    self.assertFalse(masked_spans_bounds_valid([(None, 4, 3)], 6))
    self.assertTrue(masked_spans_bounds_valid([(None, 0, 2), (None, 1, 3)], 6))
    self.assertTrue(masked_spans_bounds_valid([(None, 0, 2), (None, 2, 4)], 6))
    self.assertFalse(masked_spans_bounds_valid([(None, 0, 2), (None, 2, 5)], 6))
    self.assertFalse(masked_spans_bounds_valid([(None, 0, 0)], 0))
    self.assertTrue(masked_spans_bounds_valid([], 0))

  def test_masked_spans_overlap(self):
    self.assertFalse(masked_spans_overlap([(None, 0, 2), (None, 2, 5)]))
    self.assertTrue(masked_spans_overlap([(None, 0, 2), (None, 1, 5)]))
    self.assertFalse(masked_spans_overlap([(None, -1, -1)]))
    self.assertFalse(masked_spans_overlap([(None, 10, 10000)]))

  def test_align_char_mask_to_tokens(self):
    raise NotImplementedError()

  def test_apply_masked_spans(self):
    doc = 'She ate leftover pasta for lunch.'
    masked_spans = [
        ('ngram', 8, 14),
        ('word', 27, 5)]
    mask_type_to_substitution = {
        'ngram': '<|mask_ngram|>',
        'word': '<|mask_word|>'}
    context, answers = apply_masked_spans(
        doc,
        masked_spans,
        mask_type_to_substitution)
    self.assertEqual(context, 'She ate <|mask_ngram|> for <|mask_word|>.')
    self.assertEqual(answers, [('ngram', 'leftover pasta'), ('word', 'lunch')])

    doc = [100, 101, 102, 103, 104, 105, 106, 107]
    masked_spans = [
        ('ngram', 2, 2)]
    mask_type_to_substitution = {
        'ngram': 5000}
    context, answers = apply_masked_spans(
        doc,
        masked_spans,
        mask_type_to_substitution)
    self.assertEqual(context, [100, 101, 5000, 104, 105, 106, 107])
    self.assertEqual(answers, [('ngram', [102, 103])])


if __name__ == '__main__':
  unittest.main()
