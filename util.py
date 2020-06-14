import random

from pytorch_transformers import GPT2Tokenizer

ADDITIONAL_TYPES_TO_TOK = {
    '_DUMMY_FOR_RETRAIN_DONT_USE': '<|startoftext|>',
    'start_infill': '<|startofinfill|>',
    'end_infill': '<|endofinfill|>',
}
INFILL_TYPES_TO_TOK = {
    'paragraph': '<|infillparagraph|>',
    'sentence': '<|infillsentence|>',
    'ngram': '<|infillngram|>',
    'word': '<|infillword|>',
    'document': '<|infilldocument|>',
}
INFILL_ORDER_NUMS_TO_TOK = {}

_tokens_to_add = [ADDITIONAL_TYPES_TO_TOK[t] for t in ['_DUMMY_FOR_RETRAIN_DONT_USE', 'start_infill', 'end_infill']]
_tokens_to_add += [INFILL_TYPES_TO_TOK[t] for t in ['paragraph', 'sentence', 'ngram', 'word', 'document']]

_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
_tokenizer_orig_len = len(_tokenizer)
_tokenizer.add_special_tokens({'additional_special_tokens': _tokens_to_add})
_tokenizer_infill_len = len(_tokenizer)
_max_order_num = None

ADDITIONAL_TYPES_TO_ID = {k:_tokenizer.convert_tokens_to_ids(v) for k, v in ADDITIONAL_TYPES_TO_TOK.items()}
INFILL_TYPES_TO_ID = {k:_tokenizer.convert_tokens_to_ids(v) for k, v in INFILL_TYPES_TO_TOK.items()}
INFILL_TYPES_IDS = set(INFILL_TYPES_TO_ID.values())
INFILL_ORDER_NUMS_TO_ID = {}
SPECIAL_OUTPUT_IDS = set(list(INFILL_TYPES_IDS) + [ADDITIONAL_TYPES_TO_ID['end_infill']])

SENTENCE_SEPARATOR_IDS = [198] # \n
PARAGRAPH_SEPARATOR_IDS = [198, 198] # \n\n
#SENTENCE_SEPARATOR_IDS = [220] # <SPACE>


def get_tokenizer():
  return _tokenizer


def tokenizer_set_max_order_num(new_max_order_num):
  global _max_order_num

  if _max_order_num is None:
    assert len(_tokenizer) == _tokenizer_infill_len
    if new_max_order_num is None:
      return
    tokens = ['<|infillnum_{}|>'.format(i) for i in range(new_max_order_num)]
    _tokenizer.add_special_tokens({'additional_special_tokens': tokens})
    for i, t in enumerate(tokens):
      INFILL_ORDER_NUMS_TO_TOK[i] = t
      INFILL_ORDER_NUMS_TO_ID[i] = _tokenizer.convert_tokens_to_ids(t)
    _max_order_num = new_max_order_num
  else:
    if new_max_order_num != _max_order_num:
      raise ValueError('max_order_num cannot be changed')


def load_raw_documents(dataset_fp, style):
  with open(dataset_fp, 'r') as f:
    dataset = f.read().strip()
  documents = dataset.split('\n\n\n')
  if len(documents) == 1:
    documents = dataset.split('\n\n')
  documents = [d.strip() for d in documents]
  return documents


def split_raw_document(document_raw, style):
  document_raw = document_raw.strip()

  if document_raw == INFILL_TYPES_TO_TOK['document']:
    return INFILL_TYPES_TO_TOK['document']

  document_raw = document_raw.strip()
  if len(document_raw) == 0:
    return []

  if style == 'verse':
    paragraphs = document_raw.split('\n\n')
  elif style == 'abstract':
    paragraphs = document_raw.splitlines()
  else:
    raise ValueError()

  document_split = []
  for p in paragraphs:
    if p == INFILL_TYPES_TO_TOK['paragraph']:
      document_split.append(INFILL_TYPES_TO_TOK['paragraph'])
      continue

    if style == 'verse':
      sentences = p.strip().splitlines()
    elif style == 'abstract':
      from nltk import sent_tokenize
      sentences = sent_tokenize(p)

    p_split = []
    for s in sentences:
      if s == INFILL_TYPES_TO_TOK['sentence']:
        p_split.append(INFILL_TYPES_TO_TOK['sentence'])
        continue

      words = s.split()
      for w in words:
        if w == INFILL_TYPES_TO_TOK['document'] or w == INFILL_TYPES_TO_TOK['paragraph'] or w == INFILL_TYPES_TO_TOK['sentence']:
          raise ValueError()

      p_split.append(words)
    document_split.append(p_split)

  return document_split


def join_document(document_split, style):
  if document_split == INFILL_TYPES_TO_TOK['document']:
    return INFILL_TYPES_TO_TOK['document']

  document_joined = ''
  for i, p in enumerate(document_split):
    if i > 0:
      if style == 'verse':
        document_joined += '\n\n'
      elif style == 'abstract':
        document_joined += '\n'
      else:
        raise ValueError()

    if p == INFILL_TYPES_TO_TOK['paragraph']:
      document_joined += INFILL_TYPES_TO_TOK['paragraph']
      continue

    for j, s in enumerate(p):
      if j > 0:
        if style == 'verse':
          document_joined += '\n'
        elif style == 'abstract':
          document_joined += ' '
        else:
          raise ValueError()

      if s == INFILL_TYPES_TO_TOK['sentence']:
        document_joined += INFILL_TYPES_TO_TOK['sentence']
        continue

      document_joined += ' '.join(s)

  return document_joined


_DUMMY_WORD = 'a'
def encode_document(document_split, style):
  if document_split == INFILL_TYPES_TO_TOK['document']:
    return INFILL_TYPES_TO_ID['document']

  document_enc = []
  for i, paragraph in enumerate(document_split):
    if paragraph == INFILL_TYPES_TO_TOK['paragraph']:
      document_enc.append(INFILL_TYPES_TO_ID['paragraph'])
      continue

    paragraph_enc = []
    for j, sentence in enumerate(paragraph):
      if sentence == INFILL_TYPES_TO_TOK['sentence']:
        paragraph_enc.append(INFILL_TYPES_TO_ID['sentence'])
        continue

      sentence_enc = []
      for k, word in enumerate(sentence):
        if word == INFILL_TYPES_TO_TOK['ngram']:
          sentence_enc.append(INFILL_TYPES_TO_ID['ngram'])
          continue

        if word == INFILL_TYPES_TO_TOK['word']:
          sentence_enc.append(INFILL_TYPES_TO_ID['word'])
          continue

        if k == 0:
          if style == 'abstract' and j > 0:
            sentence_enc.append(_tokenizer.encode(_DUMMY_WORD + ' ' + word)[1:])
          else:
            sentence_enc.append(_tokenizer.encode(word))
        else:
          sentence_enc.append(_tokenizer.encode(_DUMMY_WORD + ' ' + word)[1:])
      for word_enc in sentence_enc:
        if type(word_enc) == list:
          assert len(word_enc) > 0
      paragraph_enc.append(sentence_enc)
    document_enc.append(paragraph_enc)
  return document_enc


def encode_document_no_special(document_split, style):
  document_enc = []
  for i, paragraph in enumerate(document_split):
    paragraph_enc = []
    for j, sentence in enumerate(paragraph):
      sentence_enc = []
      for k, word in enumerate(sentence):
        if k == 0:
          if style == 'abstract' and j > 0:
            sentence_enc.append(_tokenizer.encode(_DUMMY_WORD + ' ' + word)[1:])
          else:
            sentence_enc.append(_tokenizer.encode(word))
        else:
          sentence_enc.append(_tokenizer.encode(_DUMMY_WORD + ' ' + word)[1:])
      for word_enc in sentence_enc:
        assert len(word_enc) > 0
      paragraph_enc.append(sentence_enc)
    document_enc.append(paragraph_enc)
  return document_enc


def decode_document_enc(document_enc):
  if document_enc == INFILL_TYPES_TO_ID['document']:
    return INFILL_TYPES_TO_TOK['document']

  document_dec = []
  for p in document_enc:
    if p == INFILL_TYPES_TO_ID['paragraph']:
      document_dec.append(INFILL_TYPES_TO_TOK['paragraph'])
      continue

    p_dec = []
    for s in p:
      if s == INFILL_TYPES_TO_ID['sentence']:
        p_dec.append(INFILL_TYPES_TO_TOK['sentence'])
        continue

      s_dec = []
      for w in s:
        if w == INFILL_TYPES_TO_ID['ngram']:
          s_dec.append(INFILL_TYPES_TO_TOK['ngram'])
        elif w == INFILL_TYPES_TO_ID['word']:
          s_dec.append(INFILL_TYPES_TO_TOK['word'])
        else:
          s_dec.append(_tokenizer.decode(w).strip())
      p_dec.append(s_dec)
    document_dec.append(p_dec)

  return document_dec


def apply_mask(
    document_enc,
    mask_coordinates):
  if len(mask_coordinates) > 0 and mask_coordinates[0] == tuple():
    if len(mask_coordinates) > 1:
      raise ValueError()
    document_masked = INFILL_TYPES_TO_ID['document']
    document_answers = [(INFILL_TYPES_TO_ID['document'], document_enc)]
    return document_masked, document_answers

  for c in mask_coordinates:
    if len(c) > 4:
      raise ValueError()

  clen = len(mask_coordinates)
  mask_coordinates = {c[:3]:c for c in mask_coordinates}
  if len(mask_coordinates) != clen:
    raise ValueError()

  document_masked = []
  document_answers = []
  for i, p in enumerate(document_enc):
    if (i,) in mask_coordinates:
      document_masked.append(INFILL_TYPES_TO_ID['paragraph'])
      document_answers.append((INFILL_TYPES_TO_ID['paragraph'], p))
      del mask_coordinates[(i,)]
      continue

    p_masked = []
    for j, s in enumerate(p):
      if (i, j) in mask_coordinates:
        p_masked.append(INFILL_TYPES_TO_ID['sentence'])
        document_answers.append((INFILL_TYPES_TO_ID['sentence'], s))
        del mask_coordinates[(i, j)]
        continue

      s_masked = []
      k = 0
      while k < len(s):
        coord = (i, j, k)
        if coord in mask_coordinates:
          if len(mask_coordinates[coord]) == 4:
            n = mask_coordinates[coord][3]
            s_masked.append(INFILL_TYPES_TO_ID['ngram'])
            document_answers.append((INFILL_TYPES_TO_ID['ngram'], s[k:k+n]))
            k += n
          else:
            s_masked.append(INFILL_TYPES_TO_ID['word'])
            document_answers.append((INFILL_TYPES_TO_ID['word'], s[k]))
            k += 1
          del mask_coordinates[coord]
        else:
          s_masked.append(s[k])
          k += 1
      p_masked.append(s_masked)
    document_masked.append(p_masked)

  if len(mask_coordinates) > 0:
    raise ValueError()

  return document_masked, document_answers


def random_mask(
    document_enc,
    mask_document_p=0,
    mask_paragraph_p=0,
    mask_sentence_p=0,
    mask_ngram_p=0,
    mask_word_p=0,
    mask_firstword_p=0,
    mask_leadingwords_p=0,
    mask_lastword_p=0,
    mask_ngram_max_n=8):
  def trial(p):
    if p <= 0:
      return False
    else:
      return random.random() < p

  if trial(mask_document_p):
    return [tuple()]

  mask_coordinates = []
  for i, p in enumerate(document_enc):
    if trial(mask_paragraph_p):
      mask_coordinates.append((i,))
      continue

    for j, s in enumerate(p):
      assert len(s) > 0

      if trial(mask_sentence_p):
        mask_coordinates.append((i, j))
        continue

      k_start = 0
      firstword_replaced = False
      if trial(mask_firstword_p):
        mask_coordinates.append((i, j, 0))
        firstword_replaced = True
        k_start += 1

      # TODO: Figure out how to support 0 ngram here?
      if len(s) > 1 and not firstword_replaced and trial(mask_leadingwords_p):
        mask_coordinates.append((i, j, 0, len(s) - 1))
        k_start += len(s) - 1

      lastword_replaced = False
      if (len(s) > 1 or k_start == 0) and trial(mask_lastword_p):
        lastword_replaced = True
      
      # NOTE: Ensures that there are not two adjacent <|maskngram|> tokens
      ngram_okay = True
      k = k_start
      while k < len(s) - int(lastword_replaced):
        if trial(mask_word_p):
          mask_coordinates.append((i, j, k))
          ngram_okay = True
          k += 1
        elif ngram_okay and trial(mask_ngram_p):
          n = random.randint(1, min(mask_ngram_max_n, len(s) - k - int(lastword_replaced)))
          assert n > 0
          mask_coordinates.append((i, j, k, n))
          ngram_okay = False
          k += n
        else:
          ngram_okay = True
          k += 1

      if lastword_replaced:
        mask_coordinates.append((i, j, len(s) - 1))

  return mask_coordinates


def flatten_document_enc(document_enc):
  if document_enc == INFILL_TYPES_TO_ID['document']:
    return [INFILL_TYPES_TO_ID['document']]

  flat = []
  for i, p in enumerate(document_enc):
    if p == INFILL_TYPES_TO_ID['paragraph']:
      flat.append(INFILL_TYPES_TO_ID['paragraph'])
    else:
      for j, s in enumerate(p):
        if s == INFILL_TYPES_TO_ID['sentence']:
          flat.append(INFILL_TYPES_TO_ID['sentence'])
        else:
          for k, w in enumerate(s):
            if w == INFILL_TYPES_TO_ID['ngram']:
              flat.append(INFILL_TYPES_TO_ID['ngram'])
            elif w == INFILL_TYPES_TO_ID['word']:
              flat.append(INFILL_TYPES_TO_ID['word'])
            else:
              flat.extend(w)
        if j != len(p) - 1:
          flat.extend(SENTENCE_SEPARATOR_IDS)
    if i != len(document_enc) - 1:
      flat.extend(PARAGRAPH_SEPARATOR_IDS)
  return flat


def flatten_document_answers(answers):
  flat = []
  for infill_type, mask_answers in answers:
    flat.append(infill_type)
    if infill_type == INFILL_TYPES_TO_ID['document']:
      flat.extend(flatten_document_enc(mask_answers))
    elif infill_type == INFILL_TYPES_TO_ID['paragraph']:
      flat.extend(flatten_document_enc([mask_answers]))
    elif infill_type == INFILL_TYPES_TO_ID['sentence']:
      flat.extend(flatten_document_enc([[mask_answers]]))
    elif infill_type == INFILL_TYPES_TO_ID['ngram']:
      flat.extend(flatten_document_enc([[mask_answers]]))
    elif infill_type == INFILL_TYPES_TO_ID['word']:
      flat.extend(flatten_document_enc([[[mask_answers]]]))
    else:
      raise ValueError()
  return flat


def limit_answers(
    query_flat,
    answer_flat,
    limit,
    randomize=False):
  if limit is None:
    if randomize:
      raise ValueError()
    return query_flat, answer_flat

  if randomize:
    if _max_order_num is None:
      raise ValueError('Must call tokenizer_set_max_order_num first')
    if limit > _max_order_num:
      raise ValueError('Invalid limit requested')

  # Aggregate special token idxs
  query_idxs = []
  for i, t in enumerate(query_flat):
    if t in INFILL_TYPES_IDS:
      query_idxs.append(i)
  answer_idxs = []
  for i, t in enumerate(answer_flat):
    if t in INFILL_TYPES_IDS:
      answer_idxs.append(i)

  if len(query_idxs) != len(answer_idxs):
    raise ValueError()
  for qi, ai in zip(query_idxs, answer_idxs):
    if query_flat[qi] != answer_flat[ai]:
      raise ValueError()

  num_infill = len(query_idxs)
  if num_infill == 0:
    return query_flat, answer_flat

  if randomize:
    # Randomize order
    order = list(range(num_infill))
    if randomize:
      random.shuffle(order)

    # Limit order to k
    if num_infill > limit:
      order = order[:limit]

    query_segments = []
    for i in range(1, num_infill):
      query_segments.append(query_flat[query_idxs[i-1]:query_idxs[i]])
    query_segments.append(query_flat[query_idxs[-1]:])

    answer_segments = []
    for i in range(1, num_infill):
      answer_segments.append(answer_flat[answer_idxs[i-1]:answer_idxs[i]])
    answer_segments.append(answer_flat[answer_idxs[-1]:])

    # Insert special tokens
    answer_flat = []
    for oi, i in enumerate(order):
      query_segments[i].insert(0, INFILL_ORDER_NUMS_TO_ID[oi])
      answer_flat.extend(answer_segments[i])

    query_flat = []
    for q, a in zip(query_segments, answer_segments):
      query_flat.extend(q)

    return query_flat, answer_flat
  else:
    if num_infill > limit:
      last_idx = answer_idxs[limit]
      answer_flat = answer_flat[:last_idx]
    return query_flat, answer_flat
