from enum import Enum
import random

from .base import MaskFn

class MaskPunctuationType(Enum):
  SENTENCE_TERMINAL = 0
  OTHER = 1

class MaskPunctuation(MaskFn):
  def __init__(self, p=0.5):
    self.p = p

  @classmethod
  def mask_types(cls):
    return list(MaskPunctuationType)

  @classmethod
  def mask_type_serialize(cls, m_type):
    return m_type.name.lower()

  def mask(self, doc):
    masked_spans = []
    for span_offset, char in enumerate(doc):
      if not char.isalnum() and len(char.strip()) > 0 and random.random() < self.p:
        if char in ['.', '?', '!']:
          span_type = MaskPunctuationType.SENTENCE_TERMINAL
        else:
          span_type = MaskPunctuationType.OTHER
        span_len = 1
        masked_spans.append((span_type, span_offset, span_len))
    return masked_spans


from nltk import pos_tag
from ..string_util import word_tokenize
from ..tokenize_util import tokens_offsets

class MaskProperNounType(Enum):
  PROPER_NOUN = 0

class MaskProperNoun(MaskFn):
  def __init__(self, p=1.):
    try:
      pos_tag(['Ensure', 'tagger'])
    except:
      raise ValueError('Need to call nltk.download(\'averaged_perceptron_tagger\')')
    self.p = p

  @classmethod
  def mask_types(cls):
    return list(MaskProperNounType)

  @classmethod
  def mask_type_serialize(cls, m_type):
    return m_type.name.lower()

  def mask(self, doc):
    from nltk import pos_tag
    masked_spans = []
    toks = word_tokenize(doc)
    toks_offsets = tokens_offsets(doc, toks)
    toks_pos = pos_tag(toks)
    for t, off, (_, pos) in zip(toks, toks_offsets, toks_pos):
      if pos == 'NNP' and random.random() < self.p:
        masked_spans.append((MaskProperNounType.PROPER_NOUN, off, len(t)))
    return masked_spans
