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


class ResidualBottleneckHead(nn.Module):
    """Parameter-efficient residual adaptation in the CLIP embedding space.

    The last projection is initialized to zero, so the head is exactly an
    identity mapping at initialization and starts from the pretrained CLIP
    image embedding instead of replacing it with a randomly initialized head.
    """

    def __init__(self, dim=768, hidden_dim=128, dropout=0.1, residual_scale=1.0):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if residual_scale < 0.0:
            raise ValueError("residual_scale must be non-negative")

        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, hidden_dim, bias=False)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(hidden_dim, dim, bias=False)
        self.residual_scale = residual_scale
        self.reset_parameters()

    def reset_parameters(self):
        self.norm.reset_parameters()
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        residual = self.up(
            self.dropout(self.activation(self.down(self.norm(x))))
        )
        return x + self.residual_scale * residual


def _masked_similarity_graph(similarity, edge_mask, temperature):
    """Turn a sparse edge mask into a symmetric cosine-weighted graph.

    Softmax makes the total outgoing semantic and spatial edge mass comparable,
    while symmetric averaging favours reciprocal neighbours and attenuates
    one-sided, potentially noisy KNN matches.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if edge_mask.dtype != torch.bool:
        edge_mask = edge_mask.to(dtype=torch.bool)

    # finfo.min keeps all-masked rows finite. Multiplication by the mask below
    # turns such rows into zeros instead of NaNs.
    logits = (similarity / temperature).masked_fill(
        ~edge_mask, torch.finfo(similarity.dtype).min
    )
    weights = torch.softmax(logits, dim=-1) * edge_mask.to(similarity.dtype)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return 0.5 * (weights + weights.transpose(1, 2))


def _build_knn_patch_graph(patch_features, k=8, temperature=0.1):
    """Build a cosine-weighted KNN graph with k real neighbours.

    The diagonal is explicitly removed before topk. In the previous graph,
    every patch selected itself, so k=4 provided only three actual semantic
    neighbours and the later normalization added a second self-loop.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    # Rebuild the dynamic graph every forward pass, but do not backpropagate
    # through the discrete neighbour selection or its edge weights.
    x = F.normalize(patch_features.detach(), dim=-1)
    similarity = x @ x.transpose(1, 2)
    num_nodes = similarity.shape[-1]
    if num_nodes <= 1:
        return torch.zeros_like(similarity), similarity

    real_k = min(k, num_nodes - 1)
    diagonal = torch.eye(
        num_nodes,
        device=similarity.device,
        dtype=torch.bool,
    ).unsqueeze(0)
    candidate_similarity = similarity.masked_fill(
        diagonal, torch.finfo(similarity.dtype).min
    )
    neighbour_indices = candidate_similarity.topk(
        k=real_k, dim=-1
    ).indices
    edge_mask = torch.zeros_like(similarity, dtype=torch.bool)
    edge_mask.scatter_(dim=-1, index=neighbour_indices, value=True)
    adjacency = _masked_similarity_graph(
        similarity, edge_mask, temperature=temperature
    )
    return adjacency, similarity


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
        temperature=0.1,
        gate_hidden_dim=64,
        gate_source="pre_projection",
    ):
        super().__init__()
        if k <= 0:
            raise ValueError("k must be positive")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if not 0.0 < residual_weight < 1.0:
            raise ValueError("residual_weight must be in (0, 1)")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if gate_hidden_dim <= 0:
            raise ValueError("gate_hidden_dim must be positive")
        if gate_source not in {"pre_projection", "post_projection"}:
            raise ValueError(
                "gate_source must be either 'pre_projection' or "
                "'post_projection'"
            )
        self.k = k
        self.alpha = alpha
        if num_levels <= 0:
            raise ValueError("num_levels must be positive")
        self.residual_weight = residual_weight
        self.use_spatial = use_spatial
        self.num_levels = num_levels
        self.temperature = temperature
        self.gate_hidden_dim = gate_hidden_dim
        self.gate_source = gate_source
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.gate_down = nn.Linear(dim, gate_hidden_dim, bias=False)
        self.gate_activation = nn.GELU()
        self.gate_up = nn.Linear(gate_hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        """Start as stable graph smoothing with a constant residual gate."""
        nn.init.eye_(self.proj.weight)
        self.norm.reset_parameters()
        nn.init.xavier_uniform_(self.gate_down.weight)
        nn.init.zeros_(self.gate_up.weight)
        initial_gate = torch.tensor(self.residual_weight).logit().item()
        nn.init.constant_(self.gate_up.bias, initial_gate)

    def forward(self, patch_features):
        batch_size, num_nodes, _ = patch_features.shape
        semantic_adj, similarity = _build_knn_patch_graph(
            patch_features,
            k=self.k,
            temperature=self.temperature,
        )
        if self.use_spatial and self.num_levels > 1:
            # cross-level fusion: every level contributes num_nodes / num_levels
            # patches; spatial edges stay inside each level (block diagonal),
            # while the semantic graph connects across levels.
            levels_are_aligned = num_nodes % self.num_levels == 0
            nodes_per_level = num_nodes // self.num_levels
            grid_size = int(nodes_per_level ** 0.5)
            if levels_are_aligned and grid_size * grid_size == nodes_per_level:
                spatial_adj = _build_level_spatial_patch_graph(
                    batch_size,
                    grid_size,
                    self.num_levels,
                    patch_features.device,
                    semantic_adj.dtype,
                )
                spatial_adj = _masked_similarity_graph(
                    similarity,
                    spatial_adj.to(dtype=torch.bool),
                    temperature=self.temperature,
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
                spatial_adj = _masked_similarity_graph(
                    similarity,
                    spatial_adj.to(dtype=torch.bool),
                    temperature=self.temperature,
                )
                adj = self.alpha * semantic_adj + (1 - self.alpha) * spatial_adj
            else:
                adj = semantic_adj
        else:
            adj = semantic_adj
        adj = _normalize_adj(adj)

        # Work with unit-length features so the learned gate controls direction
        # mixing rather than being dominated by feature-norm differences.
        patch_features = F.normalize(patch_features, dim=-1)
        graph_message = adj @ patch_features
        graph_features = F.normalize(
            self.norm(self.proj(graph_message)), dim=-1
        )
        # Graph-V3 estimates neighbourhood reliability before the learned
        # projection. This prevents projection rotation from being conflated
        # with disagreement between a patch and its neighbours. The legacy
        # post-projection path is retained for exact Graph-V2 reproduction.
        if self.gate_source == "pre_projection":
            gate_reference = F.normalize(graph_message, dim=-1)
        else:
            gate_reference = graph_features
        disagreement = torch.abs(patch_features - gate_reference)
        gate = torch.sigmoid(
            self.gate_up(
                self.gate_activation(self.gate_down(disagreement))
            )
        )
        out = patch_features + gate * (graph_features - patch_features)
        return F.normalize(out, dim=-1)
