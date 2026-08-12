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


def _build_level_spatial_patch_graph(batch_size, grid_size, num_levels, device, dtype):
    """Spatial 8-neighbour adjacency that connects patches only within the same
    level. When several feature levels are concatenated into one big graph
    (num_levels > 1), spatial edges must not cross level boundaries, hence the
    block-diagonal layout: each level keeps its own grid, no cross-level edge."""
    height = width = grid_size
    num_nodes = height * width
    block = torch.zeros(num_nodes, num_nodes, device=device, dtype=dtype)
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
                    block[idx, nrow * width + ncol] = 1.0
    adj = torch.block_diag(*[block for _ in range(num_levels)])
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
        num_levels=1,
    ):
        super().__init__()
        self.k = k
        self.alpha = alpha
        self.residual_weight = residual_weight
        self.use_spatial = use_spatial
        self.num_levels = num_levels
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, patch_features):
        batch_size, num_nodes, _ = patch_features.shape
        semantic_adj = _build_knn_patch_graph(patch_features, k=self.k)
        if self.use_spatial and self.num_levels > 1:
            # cross-level fusion: every level contributes num_nodes / num_levels
            # patches; spatial edges stay inside each level (block diagonal),
            # while the semantic graph connects across levels.
            nodes_per_level = num_nodes // self.num_levels
            grid_size = int(nodes_per_level ** 0.5)
            if grid_size * grid_size == nodes_per_level:
                spatial_adj = _build_level_spatial_patch_graph(
                    batch_size,
                    grid_size,
                    self.num_levels,
                    patch_features.device,
                    semantic_adj.dtype,
                )
                adj = self.alpha * semantic_adj + (1 - self.alpha) * spatial_adj
            else:
                adj = semantic_adj
        elif self.use_spatial:
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
        adj = _normalize_adj(adj)
        graph_features = adj @ patch_features
        graph_features = self.norm(self.proj(graph_features))
        out = (1 - self.residual_weight) * patch_features + self.residual_weight * graph_features
        return F.normalize(out, dim=-1)
