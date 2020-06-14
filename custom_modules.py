import numpy as np
import torch
import torch.nn as nn

from transformers import GPT2Model, GPT2LMHeadModel, GPT2PreTrainedModel

NUM_UNKNOWN = 1
NUM_PHONEMES = 70 - NUM_UNKNOWN

class GPT2ModelManualEmbed(GPT2Model):
  def forward(self, input_embeds, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
    input_shape = input_embeds.size()[:2]
    if token_type_ids is not None:
      token_type_ids = token_type_ids.view(-1, input_shape[-1])
    if position_ids is not None:
      position_ids = position_ids.view(-1, input_shape[-1])

    if past is None:
      past_length = 0
      past = [None] * len(self.h)
    else:
      past_length = past[0][0].size(-2)
    if position_ids is None:
      position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=input_embeds.device)
      position_ids = position_ids.unsqueeze(0).expand(input_shape)

    # Attention mask.
    if attention_mask is not None:
      attention_mask = attention_mask.view(-1, input_shape[-1])
      # We create a 3D attention mask from a 2D tensor mask.
      # Sizes are [batch_size, 1, 1, to_seq_length]
      # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
      # this attention mask is more simple than the triangular masking of causal attention
      # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
      attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

      # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
      # masked positions, this operation will create a tensor which is 0.0 for
      # positions we want to attend and -10000.0 for masked positions.
      # Since we are adding it to the raw scores before the softmax, this is
      # effectively the same as removing these entirely.
      attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
      attention_mask = (1.0 - attention_mask) * -10000.0

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # head_mask has shape n_layer x batch x n_heads x N x N
    if head_mask is not None:
      if head_mask.dim() == 1:
        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.expand(self.config.n_layer, -1, -1, -1, -1)
      elif head_mask.dim() == 2:
        head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
      head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
    else:
      head_mask = [None] * self.config.n_layer

    inputs_embeds = input_embeds
    position_embeds = self.wpe(position_ids)
    if token_type_ids is not None:
      token_type_embeds = self.wte(token_type_ids)
    else:
      token_type_embeds = 0
    hidden_states = inputs_embeds + position_embeds + token_type_embeds
    hidden_states = self.drop(hidden_states)

    output_shape = input_shape + (hidden_states.size(-1),)

    presents = ()
    all_attentions = []
    all_hidden_states = ()
    for i, (block, layer_past) in enumerate(zip(self.h, past)):
      if self.output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

      outputs = block(hidden_states,
              layer_past=layer_past,
              attention_mask=attention_mask,
              head_mask=head_mask[i])

      hidden_states, present = outputs[:2]
      presents = presents + (present,)

      if self.output_attentions:
        all_attentions.append(outputs[2])

    hidden_states = self.ln_f(hidden_states)

    hidden_states = hidden_states.view(*output_shape)
    # Add last hidden state
    if self.output_hidden_states:
      all_hidden_states = all_hidden_states + (hidden_states,)

    outputs = (hidden_states, presents)
    if self.output_hidden_states:
      outputs = outputs + (all_hidden_states,)
    if self.output_attentions:
      # let the number of heads free (-1) so we can extract attention even after head pruning
      attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
      all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
      outputs = outputs + (all_attentions,)
    return outputs  # last hidden state, presents, (all hidden_states), (attentions)


class GPT2LMHeadUntiedModel(GPT2LMHeadModel):
  def tie_weights(self):
    print('Untied')


class GPT2LMHeadPhoneticsModel(GPT2PreTrainedModel):
  def __init__(self, config):
    super(GPT2LMHeadPhoneticsModel, self).__init__(config)

    self.transformer = GPT2ModelManualEmbed(config)
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    pronunciations = np.load('pronounce/gpt2_tokenizer_phonetics.npy')
    self.pronunciations = torch.tensor(pronunciations).long()
    self.phoneme_wte = nn.Embedding(NUM_PHONEMES, config.n_embd)
    self.phonetics_wte = nn.Embedding(config.vocab_size, config.n_embd)

    self.init_weights()
    self.tie_weights()

  def to(self, device, *args, **kwargs):
    self.pronunciations = self.pronunciations.to(device)
    return super(GPT2LMHeadPhoneticsModel, self).to(device, *args, **kwargs)

  def tie_weights(self):
    self._tie_or_clone_weights(self.lm_head,
                   self.transformer.wte)

  def resize_token_embeddings(self, new_num_tokens, *args, **kwargs):
    super(GPT2LMHeadPhoneticsModel, self).resize_token_embeddings(new_num_tokens, *args, **kwargs)

    diff = new_num_tokens - self.pronunciations.shape[0]
    if diff != 0:
      if diff < 0:
        raise NotImplementedError()
      else:
        self.pronunciations = torch.cat([self.pronunciations, torch.zeros_like(self.pronunciations[:diff])], dim=0)

    self.phonetics_wte = self._get_resized_embeddings(self.phonetics_wte, new_num_tokens)

  def forward(self, input_ids, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None):
    semantic_embeddings = self.transformer.wte(input_ids)

    unknown_phonetic_embeddings = self.phonetics_wte(input_ids)

    phoneme_wte = self.phoneme_wte.weight
    phoneme_wte = torch.cat([torch.zeros_like(phoneme_wte[:1]), phoneme_wte], dim=0)

    known_phonetic_ids = self.pronunciations[input_ids]
    known_pronunciations = known_phonetic_ids.sum(dim=2) > 0
    known_phonetic_embeddings = phoneme_wte[known_phonetic_ids]
    known_phonetic_embeddings = known_phonetic_embeddings.sum(dim=2)
    known_mask = (known_pronunciations).unsqueeze(2).float()

    phonetic_embeddings = known_mask * known_phonetic_embeddings + (1 - known_mask) * unknown_phonetic_embeddings

    input_embeddings = semantic_embeddings + phonetic_embeddings

    transformer_outputs = self.transformer(
        input_embeddings,
        past=past,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask)
    hidden_states = transformer_outputs[0]

    lm_logits = self.lm_head(hidden_states)

    outputs = (lm_logits,) + transformer_outputs[1:]
    if labels is not None:
      # Shift so that tokens < n predict n
      shift_logits = lm_logits[..., :-1, :].contiguous()
      shift_labels = labels[..., 1:].contiguous()
      # Flatten the tokens
      loss_fct = CrossEntropyLoss(ignore_index=-1)
      loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
              shift_labels.view(-1))
      outputs = (loss,) + outputs

    return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)


class GPT2LMHeadPhoneticsUntiedModel(GPT2LMHeadPhoneticsModel):
  def tie_weights(self):
    print('Untied')


class GPT2LMHeadPhoneticsKnownModel(GPT2PreTrainedModel):
  def __init__(self, config):
    super(GPT2LMHeadPhoneticsKnownModel, self).__init__(config)

    self.transformer = GPT2ModelManualEmbed(config)
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    pronunciations = np.load('pronounce/gpt2_tokenizer_phonetics.npy')
    self.pronunciations = torch.tensor(pronunciations).long()
    self.phoneme_wte = nn.Embedding(NUM_PHONEMES, config.n_embd)

    self.init_weights()
    self.tie_weights()

  def to(self, device, *args, **kwargs):
    self.pronunciations = self.pronunciations.to(device)
    return super(GPT2LMHeadPhoneticsKnownModel, self).to(device, *args, **kwargs)

  def tie_weights(self):
    self._tie_or_clone_weights(self.lm_head,
                   self.transformer.wte)

  def resize_token_embeddings(self, new_num_tokens, *args, **kwargs):
    super(GPT2LMHeadPhoneticsKnownModel, self).resize_token_embeddings(new_num_tokens, *args, **kwargs)

    diff = new_num_tokens - self.pronunciations.shape[0]
    if diff != 0:
      if diff < 0:
        raise NotImplementedError()
      else:
        self.pronunciations = torch.cat([self.pronunciations, torch.zeros_like(self.pronunciations[:diff])], dim=0)

  def forward(self, input_ids, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None):
    semantic_embeddings = self.transformer.wte(input_ids)

    phoneme_wte = self.phoneme_wte.weight
    phoneme_wte = torch.cat([torch.zeros_like(phoneme_wte[:1]), phoneme_wte], dim=0)

    known_phonetic_ids = self.pronunciations[input_ids]
    known_pronunciations = known_phonetic_ids.sum(dim=2) > 0
    known_phonetic_embeddings = phoneme_wte[known_phonetic_ids]
    known_phonetic_embeddings = known_phonetic_embeddings.sum(dim=2)

    phonetic_embeddings = known_phonetic_embeddings

    input_embeddings = semantic_embeddings + phonetic_embeddings

    transformer_outputs = self.transformer(
        input_embeddings,
        past=past,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask)
    hidden_states = transformer_outputs[0]

    lm_logits = self.lm_head(hidden_states)

    outputs = (lm_logits,) + transformer_outputs[1:]
    if labels is not None:
      # Shift so that tokens < n predict n
      shift_logits = lm_logits[..., :-1, :].contiguous()
      shift_labels = labels[..., 1:].contiguous()
      # Flatten the tokens
      loss_fct = CrossEntropyLoss(ignore_index=-1)
      loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
              shift_labels.view(-1))
      outputs = (loss,) + outputs

    return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)


class GPT2LMHeadPhoneticsUntiedKnownModel(GPT2LMHeadPhoneticsKnownModel):
  def tie_weights(self):
    print('Untied')
