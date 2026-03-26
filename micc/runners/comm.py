import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


def build_mlp(in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2):
    layers = []
    last = in_dim
    for _ in range(num_layers - 1):
        layers += [nn.Linear(last, hidden_dim), nn.ReLU()]
        last = hidden_dim
    layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)


class CommunicationEncoder(nn.Module):
    """
    m_i = p_phi(m_i | o_i)
    """
    def __init__(self, obs_dim: int, msg_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_dim, msg_dim, num_layers=2)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class AttentionAggregator(nn.Module):
    """
    M_i = Attn(q_i, {m_j}_{j != i})
    """
    def __init__(self, obs_dim: int, msg_dim: int, hidden_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.query_proj = nn.Linear(obs_dim, msg_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=msg_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.out_proj = build_mlp(msg_dim, hidden_dim, msg_dim, num_layers=2)

    def forward(self, obs_list: List[torch.Tensor], msg_list: List[torch.Tensor]) -> List[torch.Tensor]:
        n_agents = len(obs_list)
        all_msgs = torch.stack(msg_list, dim=1)  # [B, n, msg_dim]
        out = []

        for i in range(n_agents):
            if n_agents == 1:
                out.append(torch.zeros_like(msg_list[i]))
                continue

            query = self.query_proj(obs_list[i]).unsqueeze(1)  # [B,1,msg_dim]
            idx = [j for j in range(n_agents) if j != i]
            kv = all_msgs[:, idx, :]  # [B,n-1,msg_dim]
            attn_out, _ = self.attn(query, kv, kv)
            out.append(self.out_proj(attn_out.squeeze(1)))

        return out


class ActionPredictor(nn.Module):
    """
    q_eta(a_i | m_i, o_i)
    """
    def __init__(self, obs_dim: int, msg_dim: int, act_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = build_mlp(obs_dim + msg_dim, hidden_dim, act_dim, num_layers=2)

    def forward(self, obs: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, msg], dim=-1)
        return self.net(x)


def build_perturbed_messages(msg_list, noise_std: float = 0.05, dropout_prob: float = 0.1):
    out = []
    for m in msg_list:
        x = m + noise_std * torch.randn_like(m)
        mask = (torch.rand_like(x) > dropout_prob).float()
        out.append(x * mask)
    return out


def discrete_log_prob_from_onehot(logits: torch.Tensor, action_onehot: torch.Tensor):
    logp = F.log_softmax(logits, dim=-1)
    return (logp * action_onehot).sum(dim=-1, keepdim=True)