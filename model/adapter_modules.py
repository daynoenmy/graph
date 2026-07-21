import math

import torch
from torch import nn
import torch.nn.functional as F


class SimpleAdapter(nn.Module):
    def __init__(self, c_in, c_out=768):
        super(SimpleAdapter, self).__init__()
        self.fc = nn.Sequential(nn.Linear(c_in, c_out, bias=False), nn.LeakyReLU())

    def forward(self, x):
        x = self.fc(x)
        return x


class SimpleProj(nn.Module):
    def __init__(self, c_in, c_out=768, relu=True):
        super(SimpleProj, self).__init__()
        if relu:
            self.fc = nn.Sequential(nn.Linear(c_in, c_out, bias=False), nn.LeakyReLU())
        else:
            self.fc = nn.Linear(c_in, c_out, bias=False)

    def forward(self, x):
        x = self.fc(x)
        return x


def _build_knn_patch_graph(patch_features, k=8):
    x = F.normalize(patch_features, dim=-1)
    sim = x @ x.transpose(1, 2)
    k = min(max(1, k), sim.shape[-1])
    topk = sim.topk(k=k, dim=-1).indices
    adj = torch.zeros_like(sim)
    adj.scatter_(dim=-1, index=topk, value=1.0)
    adj = torch.maximum(adj, adj.transpose(1, 2))
    return adj


def _build_spatial_patch_graph(batch_size, grid_size, device, dtype):
    height = width = grid_size
    num_nodes = height * width
    adj = torch.zeros(num_nodes, num_nodes, device=device, dtype=dtype)
    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    ]
    for row in range(height):
        for col in range(width):
            idx = row * width + col
            for drow, dcol in offsets:
                nrow, ncol = row + drow, col + dcol
                if 0 <= nrow < height and 0 <= ncol < width:
                    adj[idx, nrow * width + ncol] = 1.0
    return adj.unsqueeze(0).expand(batch_size, -1, -1)


def _normalize_adj(adj):
    num_nodes = adj.shape[-1]
    eye = torch.eye(num_nodes, device=adj.device, dtype=adj.dtype).unsqueeze(0)
    adj = adj + eye
    degree = adj.sum(dim=-1).clamp(min=1e-6)
    degree_inv_sqrt = degree.pow(-0.5)
    return degree_inv_sqrt.unsqueeze(-1) * adj * degree_inv_sqrt.unsqueeze(1)


class PatchGraphBlock(nn.Module):
    def __init__(
        self,
        dim=768,
        k=8,
        alpha=0.7,
        residual_weight=0.2,
        use_spatial=True,
        feature_temperature=0.2,
        anomaly_temperature=0.2,
        soft_graph=False,
        use_spectral_norm=False,
    ):
        super().__init__()
        self.k = k
        self.alpha = alpha
        self.residual_weight = residual_weight
        self.use_spatial = use_spatial
        self.feature_temperature = feature_temperature
        self.anomaly_temperature = anomaly_temperature
        self.soft_graph = soft_graph
        projection = nn.Linear(dim, dim, bias=False)
        if use_spectral_norm:
            projection = nn.utils.parametrizations.spectral_norm(projection)
        self.proj = projection
        self.norm = nn.LayerNorm(dim)
        initial_gate = min(max(float(residual_weight), 1e-4), 1.0 - 1e-4)
        self.noise_gate_scale = nn.Parameter(torch.tensor(1.0))
        self.anomaly_gate_scale = nn.Parameter(torch.tensor(1.0))
        self.gate_bias = nn.Parameter(
            torch.tensor(math.log(initial_gate / (1.0 - initial_gate)))
        )

    @staticmethod
    def _prepare_patch_score(score, patch_features, name):
        if score is None:
            return patch_features.new_zeros(patch_features.shape[:2])
        if score.ndim == 3 and score.shape[-1] == 1:
            score = score.squeeze(-1)
        if score.shape != patch_features.shape[:2]:
            raise ValueError(
                f"{name} must have shape {patch_features.shape[:2]}, got {score.shape}"
            )
        return score.to(device=patch_features.device, dtype=patch_features.dtype)

    def forward(self, patch_features, uncertainty=None, anomaly_prob=None):
        batch_size, num_nodes, _ = patch_features.shape
        normalized_features = F.normalize(patch_features, dim=-1)
        similarity = normalized_features @ normalized_features.transpose(1, 2)
        feature_affinity = torch.exp(
            (similarity - 1.0) / max(self.feature_temperature, 1e-4)
        )
        semantic_adj = (
            feature_affinity
            if self.soft_graph
            else _build_knn_patch_graph(patch_features, k=self.k)
        )
        if self.use_spatial:
            grid_size = int(num_nodes ** 0.5)
            if grid_size * grid_size == num_nodes:
                spatial_adj = _build_spatial_patch_graph(
                    batch_size,
                    grid_size,
                    patch_features.device,
                    semantic_adj.dtype,
                )
                adj = self.alpha * semantic_adj + (1 - self.alpha) * spatial_adj
            else:
                adj = semantic_adj
        else:
            adj = semantic_adj

        # Preserve the original fixed graph behavior for baseline ablations and
        # checkpoints that do not provide a second noise view or text anchors.
        if uncertainty is None and anomaly_prob is None:
            if self.soft_graph:
                eye = torch.eye(
                    num_nodes,
                    device=patch_features.device,
                    dtype=patch_features.dtype,
                ).unsqueeze(0)
                normalized_adj = adj + eye
                normalized_adj = normalized_adj / normalized_adj.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-6)
            else:
                normalized_adj = _normalize_adj(adj)
            graph_features = normalized_adj @ patch_features
            graph_features = self.norm(self.proj(graph_features))
            out = (
                (1 - self.residual_weight) * patch_features
                + self.residual_weight * graph_features
            )
            return F.normalize(out, dim=-1)

        uncertainty = self._prepare_patch_score(
            uncertainty, patch_features, "uncertainty"
        ).detach().clamp(0.0, 1.0)
        anomaly_prob = self._prepare_patch_score(
            anomaly_prob, patch_features, "anomaly_prob"
        ).detach().clamp(0.0, 1.0)

        anomaly_difference = (
            anomaly_prob.unsqueeze(-1) - anomaly_prob.unsqueeze(1)
        ).abs()
        boundary_affinity = torch.exp(
            -anomaly_difference / max(self.anomaly_temperature, 1e-4)
        )

        # A noisy source node should contribute less, while a noisy receiver
        # may still request more information through its adaptive update gate.
        source_reliability = (1.0 - uncertainty).unsqueeze(1).clamp_min(0.05)
        if self.soft_graph:
            weighted_adj = adj * boundary_affinity * source_reliability
        else:
            weighted_adj = (
                adj * feature_affinity * boundary_affinity * source_reliability
            )
        eye = torch.eye(
            num_nodes, device=patch_features.device, dtype=patch_features.dtype
        ).unsqueeze(0)
        weighted_adj = weighted_adj + eye
        transition = weighted_adj / weighted_adj.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)

        graph_features = transition @ patch_features
        graph_features = self.norm(self.proj(graph_features))
        noise_scale = F.softplus(self.noise_gate_scale)
        anomaly_scale = F.softplus(self.anomaly_gate_scale)
        update_gate = torch.sigmoid(
            noise_scale * uncertainty
            - anomaly_scale * anomaly_prob
            + self.gate_bias
        ).unsqueeze(-1)
        out = (1.0 - update_gate) * patch_features + update_gate * graph_features
        return F.normalize(out, dim=-1)
