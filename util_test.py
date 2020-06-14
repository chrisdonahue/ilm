import random
import unittest

from util import *

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

_ABSTRACT_RAW = """
We introduce some absolute nonsense for training neural networks. It is absolutely perfect. We demonstrate that this leads to state of the art improvement on a multitude of tasks, including abstract generation.
All prior work can be disregarded. Future work needn't be conducted. ML is done. Over.
""".strip()


class TestUtil(unittest.TestCase):

  def test_split_join(self):
    verse = split_raw_document('', 'verse')
    self.assertEqual(len(verse), 0)

    verse = split_raw_document(_VERSE_RAW, 'verse')
    self.assertEqual(len(verse), 2)
    self.assertEqual(len(verse[0]), 4)
    self.assertEqual(len(verse[1]), 4)
    self.assertEqual(len(verse[0][0]), 3)
    self.assertEqual(len(verse[1][0]), 3)

    verse = join_document(verse, 'verse')
    self.assertEqual(verse, _VERSE_RAW)

    abstract = split_raw_document('', 'abstract')
    self.assertEqual(len(abstract), 0)

    abstract = split_raw_document(_ABSTRACT_RAW, 'abstract')
    self.assertEqual(len(abstract), 2)
    self.assertEqual(len(abstract[0]), 3)
    self.assertEqual(len(abstract[1]), 4)
    self.assertEqual(len(abstract[0][0]), 9)
    self.assertEqual(len(abstract[1][0]), 6)

    abstract = join_document(abstract, 'abstract')
    self.assertEqual(abstract, _ABSTRACT_RAW)


  def test_split_join_clozed(self):
    for style in ['verse', 'abstract']:
      raw = INFILL_TYPES_TO_TOK['document']
      split = split_raw_document(raw, style)
      self.assertEqual(split, INFILL_TYPES_TO_TOK['document'])
      joined = join_document(split, style)
      self.assertEqual(joined, raw)

      raw = INFILL_TYPES_TO_TOK['paragraph']
      split = split_raw_document(raw, style)
      self.assertEqual(split, [INFILL_TYPES_TO_TOK['paragraph']])
      joined = join_document(split, style)
      self.assertEqual(joined, raw)

      raw = INFILL_TYPES_TO_TOK['sentence']
      split = split_raw_document(raw, style)
      self.assertEqual(split, [[INFILL_TYPES_TO_TOK['sentence']]])
      joined = join_document(split, style)
      self.assertEqual(joined, raw)

      raw = INFILL_TYPES_TO_TOK['ngram']
      split = split_raw_document(raw, style)
      self.assertEqual(split, [[[INFILL_TYPES_TO_TOK['ngram']]]])
      joined = join_document(split, style)
      self.assertEqual(joined, raw)

      raw = INFILL_TYPES_TO_TOK['word']
      split = split_raw_document(raw, style)
      self.assertEqual(split, [[[INFILL_TYPES_TO_TOK['word']]]])
      joined = join_document(split, style)
      self.assertEqual(joined, raw)

    raw = _VERSE_RAW + '\n\n' + INFILL_TYPES_TO_TOK['paragraph']
    split = split_raw_document(raw, 'verse')
    self.assertEqual(len(split), 3)
    self.assertEqual(split[2], INFILL_TYPES_TO_TOK['paragraph'])
    self.assertEqual(join_document(split, 'verse'), raw)

    raw = _ABSTRACT_RAW + '\n' + INFILL_TYPES_TO_TOK['paragraph']
    split = split_raw_document(raw, 'abstract')
    self.assertEqual(len(split), 3)
    self.assertEqual(split[2], INFILL_TYPES_TO_TOK['paragraph'])
    self.assertEqual(join_document(split, 'abstract'), raw)


  def test_encode(self):
    for enc_fn in [encode_document, encode_document_no_special]:
      for document_raw, style in zip([_VERSE_RAW, _ABSTRACT_RAW], ['verse', 'abstract']):
        document_split = split_raw_document(document_raw, style)
        document_enc = enc_fn(document_split, style)
        document_enc_dec = decode_document_enc(document_enc)
        self.assertEqual(document_split, document_enc_dec)


  def test_encode_cloze(self):
    for style in ['verse', 'abstract']:
      tests = [
          INFILL_TYPES_TO_TOK['document'],
          INFILL_TYPES_TO_TOK['paragraph'],
          INFILL_TYPES_TO_TOK['sentence'],
          INFILL_TYPES_TO_TOK['ngram'],
          INFILL_TYPES_TO_TOK['word']
      ]

      for raw in tests:
        split = split_raw_document(raw, style)
        enc = encode_document(split, style)
        dec = decode_document_enc(enc)
        joined = join_document(dec, style)
        self.assertEqual(joined, raw)


  def test_random_mask_edges(self):
    enc = [[[[34094], [428], [7510], [523], [1327]]]]
    for i in range(10000):
      random.seed(i)
      enc_len = random.randint(1, 5)
      enc_copy = [[enc[0][0][:enc_len]]]
      mask = random_mask(
          enc_copy,
          mask_document_p=0.1,
          mask_paragraph_p=0.1,
          mask_sentence_p=0.1,
          mask_ngram_p=0.1,
          mask_firstword_p=0.1,
          mask_leadingwords_p=0.1,
          mask_lastword_p=0.1)
      enc_masked, enc_answers_tree = apply_mask(enc_copy, mask)
      enc_copy_flat = flatten_document_enc(enc_copy)
      enc_masked, enc_answers = flatten_document_enc(enc_masked), flatten_document_answers(enc_answers_tree)
      self.assertEqual(len(enc_masked + enc_answers), len(enc_copy_flat) + len(mask) * 2)
      self.assertEqual(sum(enc_masked + enc_answers), sum(enc_copy_flat) + sum([a for a, _ in enc_answers_tree]) * 2)


  def test_apply_mask(self):
    raw = _VERSE_RAW
    split = split_raw_document(raw, 'verse')
    enc = encode_document(split, 'verse')

    mask = []
    masked, answers = apply_mask(enc, mask)
    self.assertEqual(masked, enc)
    self.assertEqual(len(answers), 0)

    mask = [tuple()]
    masked, answers = apply_mask(enc, mask)
    self.assertEqual(masked, INFILL_TYPES_TO_ID['document'])
    self.assertEqual(len(answers), 1)
    self.assertEqual(answers[0][0], INFILL_TYPES_TO_ID['document'])
    self.assertEqual(answers[0][1], enc)

    mask = [(1,)]
    masked, answers = apply_mask(enc, mask)
    self.assertEqual(len(masked), 2)
    self.assertEqual(masked[0], enc[0])
    self.assertEqual(masked[1], INFILL_TYPES_TO_ID['paragraph'])
    self.assertEqual(len(answers), 1)
    self.assertEqual(answers[0][0], INFILL_TYPES_TO_ID['paragraph'])
    self.assertEqual(answers[0][1], enc[1])

    mask = [(0,1), (1,2)]
    masked, answers = apply_mask(enc, mask)
    self.assertEqual(len(masked), 2)
    self.assertEqual(masked[1][1], enc[1][1])
    self.assertEqual(masked[0][1], INFILL_TYPES_TO_ID['sentence'])
    self.assertEqual(masked[1][2], INFILL_TYPES_TO_ID['sentence'])
    self.assertEqual(len(answers), 2)
    self.assertEqual(answers[0][0], INFILL_TYPES_TO_ID['sentence'])
    self.assertEqual(answers[0][1], enc[0][1])
    self.assertEqual(answers[1][0], INFILL_TYPES_TO_ID['sentence'])
    self.assertEqual(answers[1][1], enc[1][2])

    mask = [(0,1,2), (1,2,2), (1,2,1)]
    masked, answers = apply_mask(enc, mask)
    self.assertEqual(len(masked), 2)
    self.assertEqual(masked[1][1], enc[1][1])
    self.assertEqual(masked[0][1][2], INFILL_TYPES_TO_ID['word'])
    self.assertEqual(masked[1][2][2], INFILL_TYPES_TO_ID['word'])
    self.assertEqual(masked[1][2][1], INFILL_TYPES_TO_ID['word'])
    self.assertEqual(len(answers), 3)
    self.assertEqual(answers[0][0], INFILL_TYPES_TO_ID['word'])
    self.assertEqual(answers[0][1], enc[0][1][2])
    self.assertEqual(answers[1][0], INFILL_TYPES_TO_ID['word'])
    self.assertEqual(answers[1][1], enc[1][2][1])
    self.assertEqual(answers[2][0], INFILL_TYPES_TO_ID['word'])
    self.assertEqual(answers[2][1], enc[1][2][2])

    mask = [(0,1,1,2), (1,1,1,2)]
    masked, answers = apply_mask(enc, mask)
    self.assertEqual(len(masked), 2)
    self.assertEqual(masked[1][2], enc[1][2])
    self.assertEqual(masked[0][1][1], INFILL_TYPES_TO_ID['ngram'])
    self.assertEqual(masked[1][1][1], INFILL_TYPES_TO_ID['ngram'])
    self.assertEqual(len(answers), 2)
    self.assertEqual(answers[0][0], INFILL_TYPES_TO_ID['ngram'])
    self.assertEqual(answers[0][1], enc[0][1][1:3])
    self.assertEqual(answers[1][0], INFILL_TYPES_TO_ID['ngram'])
    self.assertEqual(answers[1][1], enc[1][1][1:3])

    lines = _VERSE_RAW.splitlines()
    lines[5] = INFILL_TYPES_TO_TOK['sentence']
    verse = '\n'.join(lines)
    text_cloze = encode_document(split_raw_document(verse, 'verse'), 'verse')
    coord_cloze, coord_answers = apply_mask(encode_document(split_raw_document(_VERSE_RAW, 'verse'), 'verse'), [(1,0)])
    self.assertEqual(text_cloze, coord_cloze)
    self.assertEqual(coord_answers[0][1], enc[1][0])


  def test_mask_verse(self):
    random.seed(0)

    document = split_raw_document(_VERSE_RAW, 'verse')
    document_enc = encode_document(document, 'verse')
    document_flat = flatten_document_enc(document_enc)
    self.assertEqual(len(document_flat), 32)

    def mask_with_kwargs(mask_max_k=None, **kwargs):
      mask = random_mask(document_enc, **kwargs)
      if mask_max_k is not None:
        mask = random.sample(mask, mask_max_k)
      return apply_mask(document_enc, mask)

    document_clozed, document_answers = mask_with_kwargs()
    self.assertEqual(document_enc, document_clozed)
    document_clozed, document_answers = flatten_document_enc(document_clozed), flatten_document_answers(document_answers)
    self.assertEqual(len(document_answers), 0)
    self.assertEqual(document_clozed, document_flat)

    document_clozed, document_answers = mask_with_kwargs(mask_document_p=1.)
    document_clozed, document_answers = flatten_document_enc(document_clozed), flatten_document_answers(document_answers)
    self.assertEqual(len(document_clozed), 1)
    self.assertEqual(len(document_answers), 33)
    self.assertEqual(document_clozed[0], INFILL_TYPES_TO_ID['document'])
    self.assertEqual(document_answers[1:], document_flat)

    document_clozed, document_answers = mask_with_kwargs(mask_paragraph_p=1.)
    document_clozed, document_answers = flatten_document_enc(document_clozed), flatten_document_answers(document_answers)
    self.assertEqual(len(document_clozed), 4)
    self.assertEqual(len(document_answers), 32)
    self.assertEqual(document_clozed[0], INFILL_TYPES_TO_ID['paragraph'])
    self.assertEqual(document_clozed[-1], INFILL_TYPES_TO_ID['paragraph'])
    self.assertEqual(document_answers[0], INFILL_TYPES_TO_ID['paragraph'])
    self.assertEqual(document_answers[16], INFILL_TYPES_TO_ID['paragraph'])
    self.assertEqual(document_answers[1:16], document_flat[0:15])
    self.assertEqual(document_answers[17:], document_flat[17:])

    document_clozed, document_answers = mask_with_kwargs(mask_sentence_p=1.)
    document_clozed, document_answers = flatten_document_enc(document_clozed), flatten_document_answers(document_answers)
    self.assertEqual(len(document_clozed), 16)
    self.assertEqual(len(document_answers), 32)
    self.assertEqual(document_clozed[0], INFILL_TYPES_TO_ID['sentence'])
    self.assertEqual(document_clozed[-1], INFILL_TYPES_TO_ID['sentence'])
    self.assertEqual(sum(document_clozed), 403672)
    self.assertEqual(sum(document_answers), 542430)

    document_clozed, document_answers = mask_with_kwargs(mask_word_p=1.)
    document_clozed, document_answers = flatten_document_enc(document_clozed), flatten_document_answers(document_answers)
    self.assertEqual(len(document_clozed), 32)
    self.assertEqual(len(document_answers), 48)
    self.assertEqual(document_clozed[0], INFILL_TYPES_TO_ID['word'])
    self.assertEqual(document_clozed[-1], INFILL_TYPES_TO_ID['word'])
    self.assertEqual(sum(document_clozed), 1207896)
    self.assertEqual(sum(document_answers), 1346654)

    document_clozed, document_answers = mask_with_kwargs(mask_ngram_p=1., mask_ngram_max_n=1)
    document_clozed, document_answers = flatten_document_enc(document_clozed), flatten_document_answers(document_answers)
    self.assertEqual(len(document_clozed), 32)
    self.assertEqual(len(document_answers), 32)
    self.assertEqual(document_clozed[0], INFILL_TYPES_TO_ID['ngram'])
    self.assertEqual(document_clozed[-1], INFILL_TYPES_TO_ID['ngram'])
    self.assertEqual(sum(document_clozed), 808604)
    self.assertEqual(sum(document_answers), 941706)

    random.seed(0)
    document_clozed, document_answers = mask_with_kwargs(mask_word_p=1., mask_max_k=1)
    document_clozed, document_answers = flatten_document_enc(document_clozed), flatten_document_answers(document_answers)
    self.assertEqual(len(document_clozed), 32)
    self.assertEqual(len(document_answers), 2)
    self.assertEqual(document_clozed[21], INFILL_TYPES_TO_ID['word'])
    self.assertEqual(document_answers[0], INFILL_TYPES_TO_ID['word'])
    self.assertEqual(document_answers[1], 18565)
    self.assertEqual(sum(document_clozed), 173624)
    self.assertEqual(sum(document_answers), 68828)

    document_clozed, document_answers = mask_with_kwargs(mask_word_p=1., mask_max_k=0)
    self.assertEqual(document_enc, document_clozed)
    document_clozed, document_answers = flatten_document_enc(document_clozed), flatten_document_answers(document_answers)
    self.assertEqual(len(document_answers), 0)
    self.assertEqual(document_clozed, document_flat)


  def test_mask_abstract(self):
    random.seed(0)

    document = split_raw_document(_ABSTRACT_RAW, 'abstract')
    document_enc = encode_document(document, 'abstract')
    document_flat = flatten_document_enc(document_enc)
    self.assertEqual(len(document_flat), 65)

    def mask_with_kwargs(mask_max_k=None, **kwargs):
      mask = random_mask(document_enc, **kwargs)
      if mask_max_k is not None:
        mask = random.sample(mask, mask_max_k)
      return apply_mask(document_enc, mask)

    document_clozed, document_answers = mask_with_kwargs()
    self.assertEqual(document_enc, document_clozed)
    document_clozed, document_answers = flatten_document_enc(document_clozed), flatten_document_answers(document_answers)
    self.assertEqual(len(document_answers), 0)
    self.assertEqual(document_clozed, document_flat)

    document_clozed, document_answers = mask_with_kwargs(mask_paragraph_p=1.)
    document_clozed, document_answers = flatten_document_enc(document_clozed), flatten_document_answers(document_answers)
    self.assertEqual(len(document_clozed), 4)
    self.assertEqual(len(document_answers), 65)
    self.assertEqual(document_clozed[0], INFILL_TYPES_TO_ID['paragraph'])
    self.assertEqual(document_clozed[-1], INFILL_TYPES_TO_ID['paragraph'])
    self.assertEqual(document_answers[0], INFILL_TYPES_TO_ID['paragraph'])
    self.assertEqual(document_answers[39], INFILL_TYPES_TO_ID['paragraph'])
    self.assertEqual(document_answers[1:39], document_flat[0:38])
    self.assertEqual(document_answers[40:], document_flat[40:])

    document_clozed, document_answers = mask_with_kwargs(mask_sentence_p=1.)
    document_clozed, document_answers = flatten_document_enc(document_clozed), flatten_document_answers(document_answers)
    tokenizer = get_tokenizer()
    self.assertEqual(len(document_clozed), 14)
    self.assertEqual(len(document_answers), 65)
    self.assertEqual(document_clozed[0], INFILL_TYPES_TO_ID['sentence'])
    self.assertEqual(document_clozed[-1], INFILL_TYPES_TO_ID['sentence'])
    self.assertEqual(sum(document_clozed), 353213)
    self.assertEqual(sum(document_answers), 579459)

    document_clozed, document_answers = mask_with_kwargs(mask_word_p=1.)
    document_clozed, document_answers = flatten_document_enc(document_clozed), flatten_document_answers(document_answers)
    self.assertEqual(len(document_clozed), 54)
    self.assertEqual(len(document_answers), 105)
    self.assertEqual(document_clozed[0], INFILL_TYPES_TO_ID['word'])
    self.assertEqual(document_clozed[-1], INFILL_TYPES_TO_ID['word'])
    self.assertEqual(sum(document_clozed), 2363747)
    self.assertEqual(sum(document_answers), 2589993)

    document_clozed, document_answers = mask_with_kwargs(mask_ngram_p=1., mask_ngram_max_n=1)
    document_clozed, document_answers = flatten_document_enc(document_clozed), flatten_document_answers(document_answers)
    self.assertEqual(len(document_clozed), 58)
    self.assertEqual(len(document_answers), 59)
    self.assertEqual(document_clozed[0], INFILL_TYPES_TO_ID['ngram'])
    self.assertEqual(document_clozed[-1], INFILL_TYPES_TO_ID['ngram'])
    self.assertEqual(sum(document_clozed), 1435869)
    self.assertEqual(sum(document_answers), 1406773)


  def test_limit_answers(self):
    document = split_raw_document(_VERSE_RAW, 'verse')
    document_enc = encode_document(document, 'verse')

    orig_len = len(get_tokenizer())
    tokenizer_set_max_order_num(4)
    self.assertEqual(len(get_tokenizer()), orig_len + 4)
    tokenizer_set_max_order_num(4)
    self.assertEqual(len(get_tokenizer()), orig_len + 4)

    mask = [(0,0), (0,2), (1,0), (1,2)]
    query, answer = apply_mask(document_enc, mask)
    query_flat, answer_flat = flatten_document_enc(query), flatten_document_answers(answer)
    query_limit, answer_limit = limit_answers(query_flat, answer_flat, None)
    self.assertEqual(query_limit, query_flat)
    self.assertEqual(answer_limit, answer_flat)

    mask = []
    query, answer = apply_mask(document_enc, mask)
    query_flat, answer_flat = flatten_document_enc(query), flatten_document_answers(answer)
    query_limit, answer_limit = limit_answers(query_flat, answer_flat, None)
    self.assertEqual(query_limit, query_flat)
    self.assertEqual(answer_limit, answer_flat, [])

    mask = [(0,0), (0,2), (1,0), (1,2)]
    query, answer = apply_mask(document_enc, mask)
    query_flat, answer_flat = flatten_document_enc(query), flatten_document_answers(answer)
    query_limit, answer_limit = limit_answers(query_flat, answer_flat, 2)
    self.assertEqual(query_limit, query_flat)
    self.assertEqual(answer_limit, answer_flat[:8])

    random.seed(0)
    mask = [(0,0), (0,2), (1,0), (1,2)]
    query, answer = apply_mask(document_enc, mask)
    query_flat, answer_flat = flatten_document_enc(query), flatten_document_answers(answer)
    query_limit, answer_limit = limit_answers(query_flat, answer_flat, 2, True)
    self.assertEqual(sum(query_limit) - INFILL_ORDER_NUMS_TO_ID[0] - INFILL_ORDER_NUMS_TO_ID[1], sum(query_flat))
    self.assertEqual(query_limit[0], INFILL_ORDER_NUMS_TO_ID[1])
    self.assertEqual(query_limit[1:14], query_flat[:13])
    self.assertEqual(query_limit[14], INFILL_ORDER_NUMS_TO_ID[0])
    self.assertEqual(query_limit[15:], query_flat[13:])

    self.assertEqual(answer_limit[0], INFILL_TYPES_TO_ID['sentence'])
    self.assertEqual(answer_limit[4], INFILL_TYPES_TO_ID['sentence'])
    self.assertEqual(answer_limit[0:4], answer_flat[8:12])
    self.assertEqual(answer_limit[4:], answer_flat[:4])


if __name__ == '__main__':
  unittest.main()
