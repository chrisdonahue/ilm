import copy

import torch
import torch.nn.functional as F


def sample_from_logits(
    logits,
    temp=1.,
    topk=None,
    nucleus=1.):
  if temp == 0:
    return torch.argmax(logits, dim=-1).unsqueeze(-1)
  elif temp != 1:
    logits /= temp
  
  probs = F.softmax(logits, dim=-1)
  
  if topk is not None:
    top_probs = torch.topk(probs, topk)
    mask = F.one_hot(top_probs.indices, probs.shape[-1]).float()
    mask = mask.sum(dim=1)
    probs *= mask
    probs /= probs.sum(dim=-1)
  
  if nucleus != 1:
    probs_sorted = torch.sort(probs, descending=True, dim=-1)
    sorted_indices = probs_sorted.indices
    sorted_values = probs_sorted.values

    cumsum = torch.cumsum(sorted_values, dim=-1)
    ks = (cumsum < nucleus).long().sum(dim=-1)
    ks = torch.max(ks, torch.ones_like(ks))

    # TODO: Make this more efficient using gather
    ks = F.one_hot(ks, probs.shape[-1]).float()
    cutoffs = (sorted_values * ks).sum(-1)

    mask = (probs > cutoffs.unsqueeze(1)).float()
    probs *= mask
    
    probs /= probs.sum(keepdim=True, dim=-1)

  next_tokens = torch.multinomial(probs, num_samples=1)

  return next_tokens


def infill_with_ilm(
  model,
  special_tokens_to_ids,
  x,
  num_infills=1,
  max_sequence_length=256,
  nucleus=0.95):
  
  _sep_id = special_tokens_to_ids['<|startofinfill|>']
  _end_span_id = special_tokens_to_ids['<|endofinfill|>']
  _special_ids = special_tokens_to_ids.values()
  
  # Make sure example doesn't already ends with [sep]
  if x[-1] == _sep_id:
    x = x[:-1]
  
  # Count number of blanks
  blank_idxs = []
  for i, tok_id in enumerate(x):
    if tok_id in _special_ids:
      blank_idxs.append(i)
  k = len(blank_idxs)
  if k == 0:
    raise ValueError()
  
  # Decode until we have that many blanks
  with torch.no_grad():
    device = next(model.parameters()).device
    context = torch.tensor(x + [_sep_id], dtype=torch.long, device=device).unsqueeze(0).repeat(num_infills, 1)
    
    terminated = []

    while context.shape[0] > 0:
      logits = model(context)[0][:, -1]
      next_tokens = sample_from_logits(logits, nucleus=nucleus)
      context = torch.cat((context, next_tokens), dim=1)
      
      num_predicted_spans = (context == _end_span_id).long().sum(dim=1)
      
      terminate_expected = num_predicted_spans >= k
      terminate_toolong = torch.ones_like(context).long().sum(dim=1) >= max_sequence_length
      terminate = terminate_expected | terminate_toolong
      
      if torch.any(terminate):
        terminated_seqs = context[terminate, len(x)+1:]
        terminated.extend([list(s) for s in terminated_seqs.cpu().numpy()])
        context = context[~terminate, :]
  
  # Collect generated spans
  generated_spans = []
  for gen in terminated:
    spans = []
    while _end_span_id in gen:
      spans.append(gen[:gen.index(_end_span_id)])
      gen = gen[gen.index(_end_span_id) + 1:]
    while len(spans) < k:
      spans.append([])
    generated_spans.append(spans)
  
  # Insert into context
  generated = []
  for spans in generated_spans:
    context = copy.deepcopy(x)
    for i, j in enumerate(blank_idxs[::-1]):
      del context[j]
      context[j:j] = spans[k - 1 - i]
    generated.append(context)

  return generated
