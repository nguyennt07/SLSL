import torch
from torch import nn


class FramePredictor(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout_rate):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU()
        )
        self.predictor = nn.Linear(hidden_dim, 1)

    def forward(self, text_features):
        text_embeddings = text_features['embeddings']
        out = self.projector(text_embeddings)
        out = self.dropout(out)

        token_lengths = text_features['token_lengths'].to(torch.float32)
        token_lengths /= token_lengths.mean(dim=1, keepdim=True)

        out = self.predictor(out).squeeze() * token_lengths
        out = torch.floor(out.mean(dim=1, keepdim=True))
        return out
