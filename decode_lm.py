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


def decode_lm(
    model,
    context,
    gen_num,
    gen_length,
    stop_word=None,
    cache_attention=False,
    temp=1.,
    topk=None,
    nucleus=1.):
  if gen_num < 1:
    raise ValueError()
  if gen_length < 1:
    raise ValueError()

  device = next(model.parameters()).device

  context = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0).repeat(gen_num, 1)

  if cache_attention:
    if stop_word is not None:
      raise NotImplementedError()
    with torch.no_grad():
      logits, past = model(context)
      next_tokens = sample_from_logits(logits[:, -1], temp=temp, topk=topk, nucleus=nucleus)
      generated = next_tokens

      while generated.shape[1] < gen_length:
        logits, past = model(generated[:, -1:], past=past)
        next_tokens = sample_from_logits(logits[:, -1], temp=temp, topk=topk, nucleus=nucleus)
        generated = torch.cat((generated, next_tokens), dim=1)
  else:
    early_stopped = []
    with torch.no_grad():
      generated = context[:, :0]

      while generated.shape[1] < gen_length:
        logits, past = model(context)
        next_tokens = sample_from_logits(logits[:, -1], temp=temp, topk=topk, nucleus=nucleus)
        context = torch.cat((context, next_tokens), dim=1)
        generated = torch.cat((generated, next_tokens), dim=1)
        if stop_word is not None:
          finished = generated[:, -1] == stop_word
          if torch.any(finished):
            finished_seqs = generated[finished, :]
            early_stopped.extend(finished_seqs.cpu().numpy())
            not_finished = ~finished
            context = context[not_finished, :]
            generated = generated[not_finished, :]
            if context.shape[0] == 0:
              break

  if stop_word is not None:
    result = [list(g) for g in early_stopped]
  else:
    result = []

  if generated.shape[0] > 0:
    result += [list(g) for g in generated.cpu().numpy()]

  return result
