from functools import lru_cache

from nltk import sent_tokenize, word_tokenize as nltk_word_tokenize

from .tokenize_util import tokens_offsets

def word_tokenize(x):
  x_tokens = nltk_word_tokenize(x)
  x_tokens_offsets = tokens_offsets(x, x_tokens)
  for i, off in enumerate(x_tokens_offsets):
    if off is None and '\"' in x and (x_tokens[i] == '``' or x_tokens[i] == '\'\''):
      x_tokens[i] = '\"'
  return x_tokens


def _hierarchical_offsets_recursive(x, tokenize_fns, relative=False, parent_off=0):
  if len(tokenize_fns) == 0:
    raise ValueError()

  # Tokenize
  tokenize_fn = tokenize_fns[0]
  x_tokens = tokenize_fn(x)

  # Compute offsets and lengths
  x_tokens_offs = tokens_offsets(x, x_tokens)
  if None in x_tokens_offs:
    raise ValueError('Tokenizer produced token not found in string')
  if not relative:
    x_tokens_offs = [parent_off + t_off for t_off in x_tokens_offs]
  x_tokens_lens = [len(t) for t in x_tokens]

  if len(tokenize_fns) > 1:
    # Compute recursive offsets for tokens
    x_tokens_offs_recursive = [_hierarchical_offsets_recursive(t, tokenize_fns[1:], relative=relative, parent_off=t_off) for t, t_off in zip(x_tokens, x_tokens_offs)]
    return tuple(zip(x_tokens_offs, x_tokens_lens, x_tokens_offs_recursive))
  else:
    # Leaf
    return tuple(zip(x_tokens_offs, x_tokens_lens))


@lru_cache(maxsize=128)
def doc_to_hierarchical_offsets(d, verse=False, relative=False):
  if verse:
    tokenize_fns = [
        # Preserve original doc
        lambda d: [d],
        # Tokenize into stanzas
        lambda d: [p.strip() for p in d.split('\n\n') if len(p.strip()) > 0],
        # Tokenize into lines
        lambda p: [s.strip() for s in p.splitlines() if len(s.strip()) > 0],
        # Tokenize into words
        lambda s: [w.strip() for w in word_tokenize(s) if len(w.strip()) > 0]
    ]
  else:
    tokenize_fns = [
        # Preserve original doc
        lambda d: [d],
        # Tokenize into paragraphs
        lambda d: [p.strip() for p in d.splitlines() if len(p.strip()) > 0],
        # Tokenize into sentences
        lambda p: [s.strip() for s in sent_tokenize(p) if len(s.strip()) > 0],
        # Tokenize into words
        lambda s: [w.strip() for w in word_tokenize(s) if len(w.strip()) > 0]
    ]
  return _hierarchical_offsets_recursive(d, tokenize_fns, relative=relative)[0]
