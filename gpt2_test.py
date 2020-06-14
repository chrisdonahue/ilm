import unittest

import torch
try:
  from transformers import GPT2LMHeadModel, GPT2Tokenizer
except:
  from pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer

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


class TestGPT2(unittest.TestCase):

  def test_gpt2(self):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.to(device)

    enc = tokenizer.encode(_VERSE_RAW)
    inputs = torch.tensor(enc, dtype=torch.long, device=device).unsqueeze(0)

    model.train()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    loss, logits, activations = model(inputs, labels=inputs)
    self.assertEqual(list(logits.shape), [1, 32, 50257])
    self.assertEqual(len(activations), 12)
    for i in range(len(activations)):
      self.assertEqual(list(activations[i].shape), [2, 1, 12, 32, 64])
    self.assertAlmostEqual(loss.item(), 4.849, 3)
    self.assertAlmostEqual(logits.abs().sum().item(), 155935072.0, 1)
    self.assertAlmostEqual(activations[-1].abs().sum().item(), 42754.8, 1)

    model.resize_token_embeddings(len(tokenizer) + 4)

    loss, logits, activations = model(inputs, labels=inputs)
    self.assertEqual(list(logits.shape), [1, 32, 50261])
    self.assertEqual(len(activations), 12)
    for i in range(len(activations)):
      self.assertEqual(list(activations[i].shape), [2, 1, 12, 32, 64])
    self.assertAlmostEqual(loss.item(), 93.929, 3)
    self.assertAlmostEqual(logits.abs().sum().item(), 161710352.0, 1)
    self.assertAlmostEqual(activations[-1].abs().sum().item(), 43017.5, 1)

    model.eval()
    model.resize_token_embeddings(len(tokenizer))

    for i in range(2):
      loss, _, _ = model(inputs, labels=inputs)
      self.assertAlmostEqual(loss.item(), 3.851588, 6)


if __name__ == '__main__':
  unittest.main()
