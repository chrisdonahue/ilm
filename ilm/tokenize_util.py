from enum import Enum
from functools import lru_cache
import json
import os
import regex as re
import warnings

from .constants import GPT2_TOKENIZER_LEN
from .paths import OFFICIAL_GPT2_ENCODER_DIR
from .official_gpt2_encoder.encoder import Encoder as OfficialEncoder

class Tokenizer(Enum):
  CUSTOM = 0
  GPT2 = 1

DEFAULT_TOKENIZER = Tokenizer.GPT2


_CUSTOM_ID_TO_TOKEN = None
def set_custom_vocab_fp(vocab_fp):
  global _CUSTOM_ID_TO_TOKEN
  with open(vocab_fp, 'r') as f:
    _CUSTOM_ID_TO_TOKEN = f.read().strip().splitlines()


_TOKENIZER_TO_STATE = {}
def _get_tokenizer_state(tokenizer):
  if type(tokenizer) == str:
    try:
      tokenizer = Tokenizer[tokenizer.upper()]
    except:
      raise ValueError('Unknown tokenizer specified')

  if type(tokenizer) != Tokenizer:
    raise ValueError('Tokenizer must be from Tokenizer enum')

  if tokenizer not in _TOKENIZER_TO_STATE:
    if tokenizer == Tokenizer.GPT2:
      with open(os.path.join(OFFICIAL_GPT2_ENCODER_DIR, 'encoder.json'), 'r') as f:
        encoder_json = json.load(f)
      with open(os.path.join(OFFICIAL_GPT2_ENCODER_DIR, 'vocab.bpe'), 'r', encoding='utf-8') as f:
        bpe_data = f.read()
      bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
      official_encoder = OfficialEncoder(
          encoder=encoder_json,
          bpe_merges=bpe_merges)
      _TOKENIZER_TO_STATE[tokenizer] = official_encoder
    elif tokenizer == Tokenizer.CUSTOM:
      if _CUSTOM_ID_TO_TOKEN is None:
        raise Exception('Must call set_custom_vocab_fp first')
      CUSTOM_TOKEN_TO_ID = {v:k for k, v in enumerate(_CUSTOM_ID_TO_TOKEN)}
      if len(_CUSTOM_ID_TO_TOKEN) != len(CUSTOM_TOKEN_TO_ID):
        raise ValueError('Duplicate tokens')
      _TOKENIZER_TO_STATE[tokenizer] = (_CUSTOM_ID_TO_TOKEN, CUSTOM_TOKEN_TO_ID)
    else:
      assert False

  return _TOKENIZER_TO_STATE[tokenizer]


def update_tokenizer(additional_ids_to_tokens, tokenizer=DEFAULT_TOKENIZER):
  state = _get_tokenizer_state(tokenizer)

  additional_tokens_to_ids = {v:k for k, v in additional_ids_to_tokens.items()}
  if len(additional_ids_to_tokens) != len(additional_tokens_to_ids):
    raise ValueError()

  if tokenizer == Tokenizer.GPT2:
    vocab_size_before = len(state.encoder)
    state.encoder.update(additional_tokens_to_ids)
    state.decoder.update(additional_ids_to_tokens)
    vocab_size_after = len(state.encoder)
  elif tokenizer == Tokenizer.CUSTOM:
    raise NotImplementedError()
  else:
    assert False

  if vocab_size_after != (vocab_size_before + len(additional_ids_to_tokens)):
    raise ValueError()

  return vocab_size_after


def tokenize(s, tokenizer=DEFAULT_TOKENIZER):
  state = _get_tokenizer_state(tokenizer)
  
  if tokenizer == Tokenizer.GPT2:
    tokens_regex = re.findall(state.pat, s)
    tokens_ids = []
    for token in tokens_regex:
      token = ''.join(state.byte_encoder[b] for b in token.encode('utf-8'))
      token_ids = [state.encoder[bpe_token] for bpe_token in state.bpe(token).split(' ')]
      tokens_ids.extend(token_ids)
    raw_tokens = [state.decoder[token_id] for token_id in tokens_ids]
    tokens = [bytearray([state.byte_decoder[c] for c in token]).decode('utf-8', errors=state.errors) for token in raw_tokens]
  elif tokenizer == Tokenizer.CUSTOM:
    tokens = s.strip().split()
  else:
    assert False

  return tokens


def tokens_to_ids(tokens, tokenizer=DEFAULT_TOKENIZER):
  state = _get_tokenizer_state(tokenizer)

  if tokenizer == Tokenizer.GPT2:
    tokens_ids = []
    for token in tokens:
      token = ''.join(state.byte_encoder[b] for b in token.encode('utf-8'))
      tokens_ids.extend(state.encoder[bpe_token] for bpe_token in state.bpe(token).split(' '))
  elif tokenizer == Tokenizer.CUSTOM:
    tokens_ids = [state[1][t] for t in tokens]
  else:
    assert False

  if len(tokens_ids) != len(tokens):
    raise Exception('Token ids not equal in length to tokens')

  return tokens_ids


def ids_to_tokens(tokens_ids, tokenizer=DEFAULT_TOKENIZER):
  state = _get_tokenizer_state(tokenizer)

  if tokenizer == Tokenizer.GPT2:
    tokens = [state.decoder[token_id] for token_id in tokens_ids]
    tokens = [bytearray([state.byte_decoder[c] for c in token]).decode('utf-8', errors=state.errors) for token in tokens]
  elif tokenizer == Tokenizer.CUSTOM:
    tokens = [state[0][t] for t in tokens_ids]
  else:
    assert False

  if len(tokens) != len(tokens_ids):
    raise Exception('Tokens not equal in length to token ids')

  return tokens


def detokenize(tokens, tokenizer=DEFAULT_TOKENIZER):
  if tokenizer == Tokenizer.GPT2:
    s = ''.join(tokens)
  elif tokenizer == Tokenizer.CUSTOM:
    s = ' '.join(tokens)
  else:
    assert False

  return s


def encode(s, tokenizer=DEFAULT_TOKENIZER):
  return tokens_to_ids(tokenize(s, tokenizer=tokenizer), tokenizer=tokenizer)


def decode(tokens_ids, tokenizer=DEFAULT_TOKENIZER):
  return detokenize(ids_to_tokens(tokens_ids, tokenizer=tokenizer), tokenizer=tokenizer)


def vocab_size(tokenizer=DEFAULT_TOKENIZER):
  state = _get_tokenizer_state(tokenizer)

  if tokenizer == Tokenizer.GPT2:
    vocab_size = len(state.encoder)
  elif tokenizer == Tokenizer.CUSTOM:
    vocab_size = len(state[0])
  else:
    assert False

  return vocab_size


@lru_cache(maxsize=128)
def _tokens_offsets_and_residuals_memoized(x, x_tok):
  x_remaining_off = 0
  x_remaining = x[:]

  offsets = []
  residuals = []

  for i, t in enumerate(x_tok):
    if len(t) == 0:
      warnings.warn('Encountered empty token')

    try:
      t_off_in_x_remaining = x_remaining.index(t)
      t_res = x_remaining[:t_off_in_x_remaining]
      t_off = x_remaining_off + t_off_in_x_remaining
    except:
      t_off = None
      t_res = ''

    offsets.append(t_off)
    residuals.append(t_res)

    if t_off is not None:
      trim = t_off_in_x_remaining + len(t)
      x_remaining_off += trim
      x_remaining = x_remaining[trim:]

  rres = x_remaining

  return offsets, residuals, rres


def tokens_offsets(x, x_tok):
  if type(x_tok) != tuple:
    x_tok = tuple(x_tok)
  return _tokens_offsets_and_residuals_memoized(x, x_tok)[0]


def tokens_residuals(x, x_tok):
  if type(x_tok) != tuple:
    x_tok = tuple(x_tok)
  return _tokens_offsets_and_residuals_memoized(x, x_tok)[1:]


def align_charspan_to_tokenspan(x, x_tok, char_offset, char_len):
  if len(x_tok) == 0:
    raise ValueError()
  if char_offset < 0 or char_len < 0 or (char_offset + char_len) > len(x):
    raise ValueError()

  if type(x_tok) != tuple:
    x_tok = tuple(x_tok)
  x_tok_offsets, x_tok_residuals, x_tok_rres = _tokens_offsets_and_residuals_memoized(x, x_tok)
  if None in x_tok_offsets:
    raise ValueError()
  x_tok_residuals.append(x_tok_rres)
  x_tok_lens = [len(t) for t in x_tok]

  # Build char_idx_to_token of appropriate token for each cursor index
  # NOTE: This is one greater than len(x) because cursor can be at beginning or end.
  char_idx_to_token = [0] * len(x_tok_residuals[0])
  for i in range(len(x_tok)):
    char_idx_to_token += [i] * (x_tok_lens[i] + len(x_tok_residuals[i + 1]))
  char_idx_to_token += [len(x_tok) - 1]

  if char_len == 0:
    token_offset = char_idx_to_token[char_offset]
    token_len = 0
    char_offset = x_tok_offsets[token_offset]
    char_len = 0
  else:
    selected_x_tok = set(char_idx_to_token[char_offset:char_offset+char_len])
    token_offset = min(selected_x_tok)
    token_len = max(selected_x_tok) - token_offset + 1

    char_offset = x_tok_offsets[token_offset]
    token_end = token_offset + token_len - 1
    char_end = x_tok_offsets[token_end] + x_tok_lens[token_end]
    char_len = char_end - char_offset

  return char_offset, char_len, token_offset, token_len
