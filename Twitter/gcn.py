import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, X, A):
        """
        X: [B, N, D]
        A: [B, N, N]
        """
        AX = torch.bmm(A, X)          # [B, N, D]
        out = self.linear(AX)         # [B, N, out_dim]
        return F.relu(out)
