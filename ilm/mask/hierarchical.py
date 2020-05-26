from enum import Enum

import random

from ..string_util import doc_to_hierarchical_offsets

from .base import MaskFn

class MaskHierarchicalType(Enum):
  DOCUMENT = 0
  PARAGRAPH = 1
  SENTENCE = 2
  NGRAM = 3
  WORD = 4


class MaskHierarchical(MaskFn):
  def __init__(self, p=0.03, verse=False):
    if not verse:
      from nltk.tokenize import sent_tokenize
      try:
        sent_tokenize('Ensure punkt installed.')
      except:
        raise ValueError('Need to call nltk.download(\'punkt\')')
    self.p = p
    self.verse = verse

  @classmethod
  def mask_types(cls):
    return list(MaskHierarchicalType)

  @classmethod
  def mask_type_serialize(cls, m_type):
    return m_type.name.lower()

  def mask(
      self,
      doc,
      mask_document_p=None,
      mask_paragraph_p=None,
      mask_sentence_p=None,
      mask_word_p=None,
      mask_word_ngram_p=0.5,
      mask_word_ngram_max_length=8):
    doc_offs = doc_to_hierarchical_offsets(doc, verse=self.verse)

    mask_document_p = self.p if mask_document_p is None else mask_document_p
    mask_paragraph_p = self.p if mask_paragraph_p is None else mask_paragraph_p
    mask_sentence_p = self.p if mask_sentence_p is None else mask_sentence_p
    mask_word_p = self.p if mask_word_p is None else mask_word_p

    def _trial(p):
      if p <= 0:
        return False
      else:
        return random.random() < p

    masked_spans = []

    doc_off, doc_len, p_offs = doc_offs
    if _trial(mask_document_p):
      masked_spans.append((MaskHierarchicalType.DOCUMENT, doc_off, doc_len))
    else:
      for p_off, p_len, s_offs in p_offs:
        if _trial(mask_paragraph_p):
          masked_spans.append((MaskHierarchicalType.PARAGRAPH, p_off, p_len))
          continue

        for s_off, s_len, w_offs in s_offs:
          if _trial(mask_sentence_p):
            masked_spans.append((MaskHierarchicalType.SENTENCE, s_off, s_len))
            continue

          w_i = 0
          while w_i < len(w_offs):
            w_off, w_len = w_offs[w_i]
            if _trial(mask_word_p):
              if _trial(mask_word_ngram_p):
                # Mask ngram starting at word
                max_k = min(len(w_offs) - w_i, mask_word_ngram_max_length)
                assert max_k > 0
                k = random.randint(1, max_k)
                first_w_off = w_off
                last_w_off, last_w_len = w_offs[w_i+k-1]
                masked_spans.append((MaskHierarchicalType.NGRAM, w_off, last_w_off+last_w_len - w_off))
                w_i += k
              else:
                # Mask word
                masked_spans.append((MaskHierarchicalType.WORD, w_off, w_len))
                w_i += 1
            else:
              w_i += 1

    return masked_spans


class MaskHierarchicalVerse(MaskHierarchical):
  def __init__(self, *args, **kwargs):
    return super().__init__(*args, verse=True, **kwargs)


class MaskDocuments(MaskHierarchical):
  @classmethod
  def mask_types(cls):
    return [MaskHierarchicalType.DOCUMENT]

  def mask(
      self,
      doc,
      mask_document_p=None):
    return super().mask(
        doc,
        mask_document_p=mask_document_p,
        mask_paragraph_p=0.,
        mask_sentence_p=0.,
        mask_word_p=0.)


class MaskParagraphs(MaskHierarchical):
  @classmethod
  def mask_types(cls):
    return [MaskHierarchicalType.PARAGRAPH]

  def mask(
      self,
      doc,
      mask_paragraph_p=None):
    return super().mask(
        doc,
        mask_document_p=0.,
        mask_paragraph_p=mask_paragraph_p,
        mask_sentence_p=0.,
        mask_word_p=0.)


class MaskSentences(MaskHierarchical):
  @classmethod
  def mask_types(cls):
    return [MaskHierarchicalType.SENTENCE]

  def mask(
      self,
      doc,
      mask_sentence_p=None):
    return super().mask(
        doc,
        mask_document_p=0.,
        mask_paragraph_p=0.,
        mask_sentence_p=mask_sentence_p,
        mask_word_p=0.)


class MaskNgrams(MaskHierarchical):
  @classmethod
  def mask_types(cls):
    return [MaskHierarchicalType.NGRAM]

  def mask(
      self,
      doc,
      mask_ngram_p=None):
    return super().mask(
        doc,
        mask_document_p=0.,
        mask_paragraph_p=0.,
        mask_sentence_p=0.,
        mask_word_p=mask_ngram_p,
        mask_word_ngram_p=1.)


class MaskWords(MaskHierarchical):
  @classmethod
  def mask_types(cls):
    return [MaskHierarchicalType.WORD]

  def mask(
      self,
      doc,
      mask_word_p=None):
    return super().mask(
        doc,
        mask_document_p=0.,
        mask_paragraph_p=0.,
        mask_sentence_p=0.,
        mask_word_p=mask_word_p,
        mask_word_ngram_p=0.)


class MaskVerseDocuments(MaskDocuments):
  def __init__(self, *args, **kwargs):
    return super().__init__(*args, verse=True, **kwargs)


class MaskVerseParagraphs(MaskParagraphs):
  def __init__(self, *args, **kwargs):
    return super().__init__(*args, verse=True, **kwargs)


class MaskVerseSentences(MaskSentences):
  def __init__(self, *args, **kwargs):
    return super().__init__(*args, verse=True, **kwargs)


class MaskVerseNgrams(MaskNgrams):
  def __init__(self, *args, **kwargs):
    return super().__init__(*args, verse=True, **kwargs)


class MaskVerseWords(MaskWords):
  def __init__(self, *args, **kwargs):
    return super().__init__(*args, verse=True, **kwargs)
