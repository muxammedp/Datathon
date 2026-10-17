import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16, dropout=.05):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.base.requires_grad_(False)

    def forward(self, x):
        result = self.base(x)
        update = F.linear(F.linear(self.dropout(x.float()), self.a), self.b)
        return result + (update * self.scale).to(result.dtype)


def attach_adapters(language_model, last_layers=12, rank=8, seed=123):
    torch.manual_seed(seed)
    language_model.requires_grad_(False)
    layers = language_model.model.layers
    if not 1 <= last_layers <= len(layers):
        raise ValueError('Invalid number of adapter layers')
    for layer in layers[-last_layers:]:
        for name in ('wqkv', 'wo'):
            base = getattr(layer.attention, name)
            if not isinstance(base, nn.Linear):
                raise ValueError('Expected an unmodified InternLM2 attention projection')
            setattr(layer.attention, name, LoRALinear(base, rank=rank, alpha=rank * 2))


def adapter_state(language_model):
    return {name: value.detach().cpu().contiguous().clone()
            for name, value in language_model.named_parameters() if value.requires_grad}


def restore_adapter(language_model, state):
    parameters = {name: value for name, value in language_model.named_parameters() if value.requires_grad}
    if parameters.keys() != state.keys():
        raise ValueError('Adapter keys do not match the selected layers')
    with torch.no_grad():
        for name, value in parameters.items():
            if state[name].shape != value.shape or not torch.isfinite(state[name]).all():
                raise ValueError('Invalid adapter tensor: ' + name)
        for name, value in parameters.items():
            value.copy_(state[name].to(value.device))


def binary_logits(language_model, embeds, answer_ids):
    inputs = embeds.detach()
    if language_model.training:
        inputs.requires_grad_(True)
    hidden = language_model.model(inputs_embeds=inputs, use_cache=False,
                                  output_attentions=False, output_hidden_states=False,
                                  return_dict=True).last_hidden_state[:, -1, :]
    head = language_model.get_output_embeddings()
    ids = torch.tensor(answer_ids, device=head.weight.device)
    bias = head.bias[ids] if head.bias is not None else None
    return F.linear(hidden, head.weight[ids], bias).float()


def frozen_inputs(model, tokenizer, pixels, times, prompt):
    prefix = ''.join(f'Frame {i + 1}, time {t:.3f} seconds: <image>\n' for i, t in enumerate(times))
    template = model.conv_template.copy()
    template.system_message = model.system_message
    template.messages = []
    template.append_message(template.roles[0], prefix + prompt)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()
    replacement = '<img>' + '<IMG_CONTEXT>' * model.num_image_token + '</img>'
    query = query.replace('<image>', replacement)
    completions = [tokenizer(query + answer, return_tensors='pt')['input_ids'][0]
                   for answer in ('0', '1')]
    if len(completions[0]) != len(completions[1]):
        raise ValueError('Binary answers require matching tokenization lengths')
    if not torch.equal(completions[0][:-1], completions[1][:-1]):
        raise ValueError('Binary answers must differ only in their final token')
    answer_ids = [int(tokens[-1]) for tokens in completions]
    if answer_ids[0] == answer_ids[1]:
        raise ValueError('Binary answer tokens are identical')
    ids = completions[0][:-1].unsqueeze(0).to(pixels.device)
    selected = ids == tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
    if selected.sum().item() != len(times) * model.num_image_token:
        raise ValueError('Image token count does not match sampled frames')
    with torch.no_grad():
        visual = model.extract_feature(pixels).reshape(-1, model.config.llm_config.hidden_size)
        embeds = model.language_model.get_input_embeddings()(ids).clone()
        if len(visual) != int(selected.sum()):
            raise ValueError('Visual compression does not match the prompt')
        embeds[selected] = visual.to(embeds.dtype)
    return embeds.cpu().contiguous(), answer_ids
