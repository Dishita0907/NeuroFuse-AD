"""Compact, trainable three-branch architecture for the synthetic experiment."""
import torch
from torch import nn


class TokenEncoder(nn.Module):
    """Bidirectional contextual encoder with learned position and summary tokens."""
    def __init__(self, length, width):
        super().__init__()
        self.summary = nn.Parameter(torch.zeros(1, 1, width))
        self.position = nn.Parameter(torch.randn(1, length + 1, width) * 0.02)
        self.block = nn.TransformerEncoderLayer(width, 2, 2 * width, dropout=0,
                                                activation="gelu", batch_first=True)

    def forward(self, tokens):
        sequence = torch.cat((self.summary.expand(len(tokens), -1, -1), tokens), 1)
        return self.block(sequence + self.position[:, :sequence.size(1)])[:, 0]


class TemporalGraph(nn.Module):
    """Each visit receives a time-decayed message from the preceding visit."""
    def __init__(self, width):
        super().__init__()
        self.node = nn.Linear(2, width)
        self.message = nn.Linear(width, width, bias=False)
        self.decay = nn.Parameter(torch.zeros(width))

    def forward(self, visits):
        nodes = self.node(visits[:, :, 1:])
        gap = visits[:, 1:, :1] - visits[:, :-1, :1]
        weight = torch.exp(-gap * nn.functional.softplus(self.decay))
        messages = self.message(nodes[:, :-1]) * weight
        updated = nodes[:, 1:] + messages
        return torch.cat((nodes[:, :1], updated), 1).tanh().mean(1)


class NeuroFuse(nn.Module):
    def __init__(self, width=12):
        super().__init__()
        self.patch = nn.Conv2d(1, width, 4, stride=4)
        self.vision = TokenEncoder(16, width)
        self.embedding = nn.Embedding(8, width)
        self.language = TokenEncoder(6, width)
        self.graph = TemporalGraph(width)
        self.attention = nn.Sequential(nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))
        self.head = nn.Linear(width, 1)

    def forward(self, image, tokens, visits, available, uniform=False):
        if not available.any(1).all():
            raise ValueError("No available modality; abstain.")
        retina = self.vision(self.patch(image).flatten(2).transpose(1, 2))
        speech = self.language(self.embedding(tokens))
        features = torch.stack((retina, speech, self.graph(visits)), 1)
        logits = self.attention(features).squeeze(-1)
        if uniform:
            logits = torch.zeros_like(logits)
        weights = logits.masked_fill(~available, -torch.inf).softmax(1)
        return self.head((features * weights.unsqueeze(-1)).sum(1)).squeeze(-1), weights
