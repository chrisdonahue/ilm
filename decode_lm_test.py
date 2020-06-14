import unittest

import torch
try:
  from transformers import GPT2LMHeadModel, GPT2Tokenizer
except:
  from pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer

from decode_lm import decode_lm

_VERSE_RAW = """
We are you
Who are we
You are us
We are thee

Love is good
Love is bad
Love is tense
Love is mad
""".strip()


class TestDecodeLM(unittest.TestCase):

  def test_decode_lm(self):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()
    model.to(device)

    context = tokenizer.encode(_VERSE_RAW)
    self.assertEqual(sum(context), 141926)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    results = decode_lm(model, context, 4, 128)
    self.assertTrue(all([len(r) == 128 for r in results]))
    result_sums = [sum(r) for r in results]
    self.assertEqual(result_sums, [709083, 556013, 620042, 832863])

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    results = decode_lm(model, context, 4, 128, cache_attention=True)
    result_sums = [sum(r) for r in results]
    self.assertEqual(result_sums, [709395, 635061, 674135, 542764])


if __name__ == '__main__':
  unittest.main()
