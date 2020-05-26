import importlib

from ..tokenize_util import tokens_offsets, align_charspan_to_tokenspan

def mask_cls_str_to_type(mask_cls):
  try:
    mask_module, mask_cls = mask_cls.rsplit('.', 1)
  except:
    raise ValueError('mask_cls needs to specify both Python module and class name')
  try:
    mask_cls = getattr(importlib.import_module(mask_module), mask_cls)
  except:
    raise ValueError('Failed to import class {} from module {}'.format(mask_cls, mask_module))
  return mask_cls


def masked_spans_bounds_valid(masked_spans, d_len):
  for span_off, span_len in [s[-2:] for s in masked_spans]:
    if span_off < 0 or span_len <= 0 or span_off+span_len > d_len:
      return False
  return True


def masked_spans_overlap(masked_spans):
  last_off = None
  last_len = None
  overlap = False
  for span_off, span_len in [s[-2:] for s in sorted(masked_spans, key=lambda x: x[-2])]:
    if last_off is not None:
      if span_off < last_off + last_len:
        overlap = True
        break
    last_off = span_off
    last_len = span_len
  return overlap


def align_char_mask_to_tokens(
    d,
    d_toks,
    masked_char_spans,
    ensure_valid_bounds_in_spans=True,
    ensure_nonoverlapping_spans=True):
  # Find token offsets in characters
  try:
    d_toks_offs = tokens_offsets(d, d_toks)
    assert None not in d_toks_offs
  except:
    raise ValueError('Tokens could not be aligned to document')

  # Align character spans to model tokens
  masked_token_spans = [align_charspan_to_tokenspan(d, d_toks, char_off, char_len)[2:] for t, char_off, char_len in masked_char_spans]

  if ensure_valid_bounds_in_spans and not masked_spans_bounds_valid(masked_token_spans, len(d_toks)):
    raise ValueError('Alignment produced invalid token spans')
  if ensure_nonoverlapping_spans and masked_spans_overlap(masked_token_spans):
    raise ValueError('Alignment produced overlapping token spans')

  # TODO: Not strict? Just remove invalid char masks?

  result = []
  for (char_t, char_off, char_len), (tok_off, tok_len) in zip(masked_char_spans, masked_token_spans):
    # Token span must contain strictly more text than the original span
    orig_span = d[char_off:char_off+char_len]
    tok_span = ''.join(d_toks[tok_off:tok_off+tok_len])
    if orig_span not in tok_span:
      raise Exception('Failed to align character span to tokens')

    result.append((char_t, tok_off, tok_len))

  return result


def _apply_masked_spans(
    doc,
    masked_spans,
    mask_type_to_substitution):
  if None in doc:
    raise ValueError()

  context = doc[:]
  answers = []
  for (span_type, span_off, span_len) in masked_spans:
    if span_len == 0:
      continue

    if span_off >= len(context):
      raise ValueError()

    answers.append((span_type, context[span_off:span_off+span_len]))
    context[span_off:span_off+span_len] = [None] * span_len

  for (_, span) in answers:
    if None in span:
      raise ValueError('Overlapping mask detected')

  for i, (span_type, _, span_len) in enumerate(masked_spans):
    span_off = context.index(None)
    assert all([i is None for i in context[span_off:span_off+span_len]])
    del context[span_off:span_off+span_len]
    substitution = mask_type_to_substitution[span_type]
    if type(substitution) == list:
      context[span_off:span_off] = substitution
    else:
      context.insert(span_off, substitution)
  assert None not in context

  return context, answers


def apply_masked_spans(
    doc_str_or_token_list,
    masked_spans,
    mask_type_to_substitution):
  if type(doc_str_or_token_list) == str:
    context, answers = _apply_masked_spans(
        list(doc_str_or_token_list),
        masked_spans,
        {k:list(v) for k, v in mask_type_to_substitution.items()})
    context = ''.join(context)
    answers = [(t, ''.join(s)) for t, s in answers]
    return context, answers
  elif type(doc_str_or_token_list) == list:
    return _apply_masked_spans(
        doc_str_or_token_list,
        masked_spans,
        mask_type_to_substitution)
  else:
    raise ValueError()
