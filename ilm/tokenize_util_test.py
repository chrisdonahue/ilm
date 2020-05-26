import tempfile
import unittest

from transformers import GPT2Tokenizer

from ilm.datasets import roc_stories
from ilm.tokenize_util import *

class TestTokenizeUtil(unittest.TestCase):

  def test_tokenize_custom(self):
    with tempfile.NamedTemporaryFile() as f:
      with open(f.name, 'w') as f:
        f.write('\n'.join(['a', 'b', 'c', 'd']))
      set_custom_vocab_fp(f.name)
    expected_s = 'a b c d c b a'
    expected_tokens = ['a', 'b', 'c', 'd', 'c', 'b', 'a']
    expected_tokens_ids = [0, 1, 2, 3, 2, 1, 0]
    self.assertEqual(
        tokenize(expected_s, Tokenizer.CUSTOM),
        expected_tokens)
    self.assertEqual(
        tokens_to_ids(expected_tokens, Tokenizer.CUSTOM),
        expected_tokens_ids)
    self.assertEqual(
        ids_to_tokens(expected_tokens_ids, Tokenizer.CUSTOM),
        expected_tokens)
    self.assertEqual(
        detokenize(expected_tokens, Tokenizer.CUSTOM),
        expected_s)
    self.assertEqual(
        encode(expected_s, Tokenizer.CUSTOM),
        expected_tokens_ids)
    self.assertEqual(
        decode(expected_tokens_ids, Tokenizer.CUSTOM),
        expected_s)

  def test_tokenize(self):
    examples = [
        ' This is an example.   It is cool '
    ]
    expected_tokens = [
        [' This', ' is', ' an', ' example', '.', ' ', ' ', ' It', ' is', ' cool', ' ']
    ]
    expected_ids = [
        [770, 318, 281, 1672, 13, 220, 220, 632, 318, 3608, 220]
    ]
    for ex, expected_ex_tokens, expected_ex_ids in zip(examples, expected_tokens, expected_ids):
      ex_tokens = tokenize(ex)
      self.assertEqual(ex_tokens, expected_ex_tokens)

      ex_tokens_off = tokens_offsets(ex, ex_tokens)
      self.assertTrue(all([t_off is not None for t_off in ex_tokens_off]))

      ex_ids = tokens_to_ids(ex_tokens)
      self.assertEqual(ex_ids, expected_ex_ids)
      self.assertEqual(len(ex_tokens), len(ex_ids))

      ex_tokens_hat = ids_to_tokens(ex_ids)
      self.assertEqual(ex_tokens, ex_tokens_hat)

      ex_hat = detokenize(ex_tokens_hat)
      self.assertEqual(ex_hat, ex)

    ref_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    for s in roc_stories(split='train', exclude_nonstandard=False)[:1000]:
      s_tokens = tokenize(s)

      s_tokens_off = tokens_offsets(s, s_tokens)
      self.assertTrue(all([t_off is not None for t_off in s_tokens_off]))

      s_ids = tokens_to_ids(s_tokens)
      self.assertEqual(len(s_tokens), len(s_ids))

      s_ids_enc = encode(s)
      self.assertEqual(s_ids_enc, s_ids)

      s_ids_ref = ref_tokenizer.encode(s)
      self.assertEqual(s_ids, s_ids_ref)

      s_tokens_hat = ids_to_tokens(s_ids)
      self.assertEqual(s_tokens, s_tokens_hat)

      s_hat = detokenize(s_tokens_hat)
      self.assertEqual(s_hat, s)

      s_hat_dec = decode(s_ids)
      self.assertEqual(s_hat_dec, s)

  def test_tokens_offsets_and_resdiuals(self):
    examples = [
        ' This is an example.  ',
        ' Please tokenize and  align this   correctly :) \n',
    ]
    examples_tokens = [
        ['This', 'notarealtoken', 'an', 'example'],
        tokenize(examples[1]),
    ]
    expected_token_offsets = [
        [1, None, 9, 12],
        [0, 7, 13, 16, 20, 21, 27, 32, 33, 34, 44, 47, 48],
    ]
    expected_token_residuals = [
        [' ', '', ' is ', ' ', '.  '],
        [''] * (len(examples_tokens[1]) + 1)
    ]

    for ex, ex_tokens, expected_off, expected_res in zip(
        examples,
        examples_tokens,
        expected_token_offsets,
        expected_token_residuals):
      ex_tokens_offs = tokens_offsets(ex, ex_tokens)
      self.assertEqual(ex_tokens_offs, expected_off)

      ex_tokens_lres, ex_rres = tokens_residuals(ex, ex_tokens)
      ex_tokens_res = ex_tokens_lres + [ex_rres]
      self.assertEqual(ex_tokens_res, expected_res)

      if None not in expected_off:
        tok_len = sum([len(t) for t in ex_tokens])
        res_len = sum([len(r) for r in ex_tokens_res])
        self.assertEqual(tok_len + res_len, len(ex))

  def test_align_charspan_to_tokenspan(self):
    sentence = ' This  is a   sentence. '
    tokens = ['This', 'is', 'a', 'sentence']

    answers = '0000000111222233333333333'
    for i in range(len(sentence) + 1):
      self.assertEqual(
          align_charspan_to_tokenspan(sentence, tokens, i, 0)[2:],
          (int(answers[i]), 0))

    self.assertEqual(
        align_charspan_to_tokenspan(sentence, tokens, 0, 7),
        (1, 4, 0, 1))
    self.assertEqual(
        align_charspan_to_tokenspan(sentence, tokens, 0, 8),
        (1, 8, 0, 2))
    self.assertEqual(
        align_charspan_to_tokenspan(sentence, tokens, 7, 1),
        (7, 2, 1, 1))
    self.assertEqual(
        align_charspan_to_tokenspan(sentence, tokens, 0, 24),
        (1, 21, 0, 4))
    self.assertEqual(
        align_charspan_to_tokenspan(sentence, tokens, 23, 1),
        (14, 8, 3, 1))

    sentence = 'This is preposterous'
    tokens = ['This', 'is', 'preposterous']
    gpt2_tokens = tokenize(sentence)
    self.assertEqual(len(tokens), 3)
    self.assertEqual(len(gpt2_tokens), 5)
    sel_off, sel_len, _, _, = align_charspan_to_tokenspan(sentence, tokens, 10, 2)
    self.assertEqual(sentence[sel_off:sel_off+sel_len], 'preposterous')
    _, _, gpt2_tokens_off, gpt2_tokens_len = align_charspan_to_tokenspan(
        sentence,
        gpt2_tokens,
        sel_off, sel_len)
    self.assertEqual(gpt2_tokens_off, 2)
    self.assertEqual(gpt2_tokens_len, 3)

  def test_update_tokenizer(self):
    ids_to_tokens = {
        50257: '<|hrm|>',
        50258: '<|hrmmmm|>'
    }
    vocab_size = update_tokenizer(ids_to_tokens)
    self.assertEqual(vocab_size, 50259)
    self.assertEqual(decode([15496, 995, 50256, 50257, 50258]), 'Hello world<|endoftext|><|hrm|><|hrmmmm|>')
    #TODO: This does not work for now... do we need it to work ever?
    #self.assertEqual(encode('<|endoftext|><|hrm|><|hrmmmm|>'), [50256, 50257, 50258])


if __name__ == '__main__':
  unittest.main()
