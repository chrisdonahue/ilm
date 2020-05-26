from collections import Counter
import random
import unittest

import ilm.tokenize_util

from ilm.datasets import roc_stories
from ilm.mask.util import mask_cls_str_to_type
from create_ilm_examples import *
from train_ilm import *

class TestTrainIlm(unittest.TestCase):

  def test_doc_and_char_masks_to_input_and_tt(self):
    masker = mask_cls_str_to_type('ilm.mask.hierarchical.MaskHierarchical')()
    tokenizer = ilm.tokenize_util.Tokenizer.GPT2
    start_infill_id = 51000
    end_infill_id = 51001
    mask_type_to_id = {t:51002+i for i, t in enumerate(masker.mask_types())}
    mask_id_to_str = {
      start_infill_id: '<|startofinfill|>',
      end_infill_id: '<|endofinfill|>',
    }
    mask_id_to_str.update({mask_type_to_id[t]:'<|infill_{}|>'.format(masker.mask_type_serialize(t)) for t in masker.mask_types()})
    word_type = masker.mask_types()[-1]
    ilm.tokenize_util.update_tokenizer(mask_id_to_str, tokenizer)

    task_to_expected = {
        Task.ILM: [
          ('She', TargetType.CONTEXT),
          (' ate', TargetType.CONTEXT),
          ('<|infill_word|>', TargetType.CONTEXT_SPECIAL),
          (' for', TargetType.CONTEXT),
          ('<|infill_word|>', TargetType.CONTEXT_SPECIAL),
          ('!', TargetType.CONTEXT),
          ('<|startofinfill|>', TargetType.CONTEXT_INFILL_SEP),
          (' cereal', TargetType.INFILL),
          ('<|endofinfill|>', TargetType.INFILL_SPECIAL),
          (' breakfast', TargetType.INFILL),
          (' this', TargetType.INFILL),
          (' morning', TargetType.INFILL),
          ('<|endofinfill|>', TargetType.INFILL_SPECIAL)
        ],
        Task.NO_CONTEXT_ILM: [
          ('<|startofinfill|>', TargetType.CONTEXT_INFILL_SEP),
          (' cereal', TargetType.INFILL),
          ('<|endofinfill|>', TargetType.INFILL_SPECIAL),
          (' breakfast', TargetType.INFILL),
          (' this', TargetType.INFILL),
          (' morning', TargetType.INFILL),
          ('<|endofinfill|>', TargetType.INFILL_SPECIAL)
        ],
        Task.NAIVE: [
          ('She', TargetType.CONTEXT),
          (' ate', TargetType.CONTEXT),
          ('<|infill_word|>', TargetType.CONTEXT_SPECIAL),
          (' for', TargetType.CONTEXT),
          ('<|infill_word|>', TargetType.CONTEXT_SPECIAL),
          ('!', TargetType.CONTEXT),
          ('<|startofinfill|>', TargetType.CONTEXT_INFILL_SEP),
          ('She', TargetType.INFILL_REDUNDANT),
          (' ate', TargetType.INFILL_REDUNDANT),
          (' cereal', TargetType.INFILL),
          (' for', TargetType.INFILL_SPECIAL),
          (' breakfast', TargetType.INFILL),
          (' this', TargetType.INFILL),
          (' morning', TargetType.INFILL),
          ('!', TargetType.INFILL_SPECIAL),
          ('<|endofinfill|>', TargetType.INFILL_REDUNDANT)
        ],
        Task.LM: [
          ('<|startofinfill|>', TargetType.CONTEXT_INFILL_SEP),
          ('She', TargetType.INFILL_REDUNDANT),
          (' ate', TargetType.INFILL_REDUNDANT),
          (' cereal', TargetType.INFILL),
          (' for', TargetType.INFILL_SPECIAL),
          (' breakfast', TargetType.INFILL),
          (' this', TargetType.INFILL),
          (' morning', TargetType.INFILL),
          ('!', TargetType.INFILL_SPECIAL),
          ('<|endofinfill|>', TargetType.INFILL_REDUNDANT)
        ],
        Task.REVERSE_LM: [
          ('<|startofinfill|>', TargetType.CONTEXT_INFILL_SEP),
          ('!', TargetType.INFILL_REDUNDANT),
          (' morning', TargetType.INFILL),
          (' this', TargetType.INFILL),
          (' breakfast', TargetType.INFILL),
          (' for', TargetType.INFILL_SPECIAL),
          (' cereal', TargetType.INFILL),
          (' ate', TargetType.INFILL_SPECIAL),
          ('She', TargetType.INFILL_REDUNDANT),
          ('<|endofinfill|>', TargetType.INFILL_REDUNDANT)
        ],
    }

    tasks = list(Task)
    
    doc = 'She ate cereal for breakfast this morning'
    char_masks = [[
        (word_type, doc.index('cereal'), len('cereal')),
        (word_type, doc.index('breakfast this morning'), len('breakfast this morning'))]]
    for d in [doc, doc + '!']:
      for task in list(Task):
        expected = task_to_expected[task]
        if '!' not in d:
          expected = [(a, b) for a, b in expected if a != '!']
          if task in [Task.NAIVE, Task.LM]:
            expected[-1] = ('<|endofinfill|>', TargetType.INFILL_SPECIAL)
        for sequence_length in range(16):
          inp, tt = doc_and_char_masks_to_input_and_tt(
              d,
              char_masks,
              tokenizer,
              start_infill_id,
              end_infill_id,
              mask_type_to_id,
              task,
              sequence_length)
          self.assertEqual(inp.shape, (1, sequence_length))
          self.assertEqual(tt.shape, (1, sequence_length))

          inp = list(inp[0])
          tt = list(tt[0])
          try:
            inp_len = tt.index(0)
          except:
            inp_len = len(inp)

          self.assertTrue(inp_len <= len(expected))
          self.assertTrue(np.array_equal(inp[inp_len:], np.zeros_like(inp[inp_len:])))
          self.assertTrue(np.array_equal(tt[inp_len:], np.zeros_like(tt[inp_len:])))

          inp = inp[:inp_len]
          tt = tt[:inp_len]

          toks = ilm.tokenize_util.ids_to_tokens(inp)
          tt = [TargetType(t) for t in tt]

          self.assertEqual(list(zip(toks, tt)), expected[:sequence_length])

    # Make dataset
    random.seed(0)
    docs = roc_stories('valid')
    docs_masked, errors = randomly_mask_dataset(
        docs,
        masker,
        16,
        64)
    task_to_expected_count = {
        # NOTE: There are 54859 total masks; 29 of them fail to apply so 54830
        # PAD, CONTEXT, CONTEXT_SPECIAL, CONTEXT_INFILL_SEP, INFILL, INFILL_SPECIAL, INFILL_REDUNDANT
        Task.ILM: [5899898, 1368853, 54830, 29849, 233084, 54830, 0],
        Task.NO_CONTEXT_ILM: [7323581, 0, 0, 29849, 233084, 54830, 0],
        Task.NAIVE: [4556026, 1368853, 54830, 29849, 233084, 53082, 1345620],
        Task.LM: [5979709, 0, 0, 29849, 233084, 53082, 1345620],
        Task.REVERSE_LM: [5979709, 0, 0, 29849, 233084, 53082, 1345620]
    }
    for task in list(Task):
      tt_to_count = Counter()
      num_masks_total = 0
      for doc, char_masks in docs_masked:
        _, tts = doc_and_char_masks_to_input_and_tt(
            doc,
            char_masks,
            tokenizer,
            start_infill_id,
            end_infill_id,
            mask_type_to_id,
            task,
            256)
        for k, c in {TargetType(k):v for k, v in zip(*np.unique(tts, return_counts=True))}.items():
          tt_to_count[k] += c
        num_masks_total += sum([len(c) for c in char_masks])
      print('-' * 80)
      print(task)
      print(num_masks_total)
      for k, c in tt_to_count.items():
        print('{}: {}'.format(k, c))
      self.assertEqual([tt_to_count[t] for t in TargetType], task_to_expected_count[task])


if __name__ == '__main__':
  unittest.main()
