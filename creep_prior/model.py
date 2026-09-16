"""Small conditional attention model for synthetic-task pretraining.

Inputs contain only observed context values, properties and future query times.
Each query independently cross-attends to the encoded context. No future targets
or other query representations enter that attention's keys/values.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class CreepPFN(nn.Module):
    def __init__(self, width=64, layers=2, heads=4, dropout=0.05,
                 self_attention=True, cross_attention=True, use_properties=True,
                 use_query_gap=True, query_skip=True, query_mlp=True,
                 learn_scale=True):
        super().__init__()
        if width % heads:
            raise ValueError('Width must be divisible by attention heads')
        self.config = dict(width=width, layers=layers, heads=heads, dropout=dropout,
                           self_attention=self_attention, cross_attention=cross_attention,
                           use_properties=use_properties, use_query_gap=use_query_gap,
                           query_skip=query_skip, query_mlp=query_mlp,
                           learn_scale=learn_scale)
        self.self_attention = self_attention
        self.cross_attention_enabled = cross_attention
        self.use_properties = use_properties
        self.use_query_gap = use_query_gap
        self.query_skip = query_skip
        self.learn_scale = learn_scale
        self.context_projection = nn.Linear(3, width)
        self.property_projection = nn.Sequential(nn.Linear(6, width), nn.GELU(), nn.Linear(width, width))
        if self_attention:
            block = nn.TransformerEncoderLayer(width, heads, 3 * width, dropout,
                                               activation="gelu", batch_first=True, norm_first=True)
            self.context_encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        else:
            # Retain token-wise feed-forward processing while removing
            # all context-to-context attention.
            self.context_encoder = nn.ModuleList([nn.Sequential(
                nn.LayerNorm(width), nn.Linear(width, 3 * width), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(3 * width, width), nn.Dropout(dropout))
                for _ in range(layers)])
        self.context_norm = nn.LayerNorm(width)
        self.query_projection = (nn.Sequential(nn.Linear(4, width), nn.GELU(), nn.Linear(width, width))
                                 if query_mlp else nn.Linear(4, width))
        if cross_attention:
            self.cross_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2))
        nn.init.zeros_(self.output[-1].weight)
        with torch.no_grad():
            self.output[-1].bias.copy_(torch.tensor([0., -1.]))

    def forward(self, context_times, context_values, context_mask, query_times, features):
        mask = context_mask.bool()
        # Sanitize masked positions before deriving any statistics or embeddings.
        t = torch.where(mask, context_times, 0.)
        y = torch.where(mask, context_values, 0.)
        scale = y.abs().amax(dim=1).clamp_min(1.)
        last_index = t.masked_fill(~mask, -1.).argmax(dim=1)
        rows = torch.arange(len(t), device=t.device)
        last_t, last_y = t[rows, last_index], y[rows, last_index]
        y_scaled = torch.asinh(y / scale[:, None])
        time_scale = math.log1p(160.)
        context = torch.stack([torch.log1p(t) / time_scale, t / 160., y_scaled], dim=-1)
        if self.use_properties:
            props = self.property_projection(torch.cat([features, torch.log1p(scale)[:, None]], dim=-1))
        else:
            props = context.new_zeros((len(t), self.context_projection.out_features))
        encoded = self.context_projection(context) + props[:, None]
        if self.self_attention:
            encoded = self.context_encoder(encoded, src_key_padding_mask=~mask)
        else:
            for block in self.context_encoder:
                encoded = encoded + block(encoded)
        encoded = self.context_norm(encoded)
        baseline = torch.asinh(last_y / scale)
        gap = torch.log1p((query_times - last_t[:, None]).clamp_min(0)) / time_scale
        if not self.use_query_gap:
            gap = torch.zeros_like(gap)
        query = torch.stack([torch.log1p(query_times) / time_scale, query_times / 160.,
                             gap,
                             baseline[:, None].expand_as(query_times)], dim=-1)
        query = self.query_projection(query) + props[:, None]
        if self.cross_attention_enabled:
            attended, _ = self.cross_attention(query, encoded, encoded, key_padding_mask=~mask, need_weights=False)
        else:
            attended = (encoded * mask[..., None]).sum(dim=1) / mask.sum(dim=1).clamp_min(1)[:, None]
            attended = attended[:, None].expand_as(query)
        out = self.output(query + attended if self.query_skip else attended)
        mu = baseline[:, None] + out[..., 0]
        sigma = F.softplus(out[..., 1]) + .03 if self.learn_scale else torch.full_like(mu, .34)
        return mu, sigma, scale


def task_nll(mu, sigma, scale, targets, target_mask):
    """Task-balanced Gaussian NLL in asinh(increment/context_scale) space."""
    values = torch.where(target_mask, targets, 0.)
    z = torch.asinh(values / scale[:, None])
    point_loss = .5 * ((z - mu) / sigma).square() + sigma.log() + .5 * math.log(2 * math.pi)
    return ((point_loss * target_mask).sum(1) / target_mask.sum(1).clamp_min(1)).mean()


def predictive_quantiles(mu, sigma, scale):
    """Median and central 90% marginal interval in original compliance units."""
    z90 = 1.6448536269514722
    return tuple(torch.sinh(value) * scale[:, None] for value in
                 (mu, mu - z90 * sigma, mu + z90 * sigma))
