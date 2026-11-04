import torch
import torch.nn as nn
from enum import Enum
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool


class ActivationFunction(Enum):
    silu = nn.SiLU
    sigmoid = nn.Sigmoid
    tanh = nn.Tanh
    softplus = nn.Softplus
    relu = nn.ReLU
    leakyrelu = nn.LeakyReLU
    prelu = nn.PReLU
    selu = nn.SELU
    elu = nn.ELU


class RegressionHead(nn.Module):
    """
    普通 MLP 回归头，替代原来的 MoE。
    """
    def __init__(self, in_feature, hidden_dim, dropout_rate, activation_type):
        super().__init__()
        activation = ActivationFunction[activation_type].value()

        self.net = nn.Sequential(
            nn.Linear(in_feature, hidden_dim, bias=True),
            activation,
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim, hidden_dim // 2, bias=True),
            activation,
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim // 2, 1, bias=True),
        )

    def forward(self, x):
        return self.net(x)


class dmpnn(nn.Module):
    def __init__(
        self,
        node_feat_dim,
        edge_feat_dim,
        edge_output_dim,
        node_output_dim,
        extra_dim,
        num_rounds,
        dropout_rate,
        activation_type1,
        activation_type2,
        num_experts,   # 为兼容你现有 train.py，保留这个参数但不再使用
        moe_hid_dim,   # 这里把它作为 head hidden dim 使用
        pooling="mean",  # Legacy checkpoints may omit this argument.
    ):
        super().__init__()

        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.edge_output_dim = edge_output_dim
        self.node_output_dim = node_output_dim
        self.extra_dim = extra_dim
        self.num_rounds = num_rounds
        self.pooling = pooling

        if pooling not in {"mean", "mean_max", "mean_sum_max"}:
            raise ValueError(
                f"Unsupported pooling={pooling!r}; expected 'mean', 'mean_max', or 'mean_sum_max'."
            )

        self.activation = ActivationFunction[activation_type1].value()
        self.dropout = nn.Dropout(dropout_rate)

        # Edge initialization: concat(source node feat, bond feat) -> hidden edge state
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_feat_dim + edge_feat_dim, edge_output_dim, bias=False),
            self.activation,
        )

        # Edge message update
        self.edge_update_mlp = nn.Sequential(
            nn.Linear(edge_output_dim, edge_output_dim, bias=False)
        )

        # Node readout/update
        self.node_mlp = nn.Sequential(
            nn.Linear(node_feat_dim + edge_output_dim, node_output_dim, bias=True),
            self.activation,
            self.dropout,
        )

        graph_feature_dim = node_output_dim * {"mean": 1, "mean_max": 2, "mean_sum_max": 3}[pooling]

        # 普通回归头，替代 MoE
        self.head = RegressionHead(
            in_feature=graph_feature_dim + extra_dim,
            hidden_dim=moe_hid_dim,
            dropout_rate=dropout_rate,
            activation_type=activation_type2,
        )

    @staticmethod
    def _scatter_add(src_tensor: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:

        out = torch.zeros(
            num_nodes,
            src_tensor.size(1),
            dtype=src_tensor.dtype,
            device=src_tensor.device
        )
        out.scatter_add_(0, index.unsqueeze(1).expand_as(src_tensor), src_tensor)
        return out

    def setup_reverse_edges(
        self, edge_index: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        """Return reverse edges for the paired order created by data.py."""
        del num_nodes
        num_edges = edge_index.size(1)
        if num_edges % 2 != 0:
            raise ValueError("Directed edge count must be even (forward/reverse pairs).")
        edge_ids = torch.arange(num_edges, device=edge_index.device)
        return torch.bitwise_xor(edge_ids, 1)

    def _pool_graph(self, node_h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        mean_features = global_mean_pool(node_h, batch)
        if self.pooling == "mean":
            return mean_features
        if self.pooling == "mean_max":
            return torch.cat([mean_features, global_max_pool(node_h, batch)], dim=-1)
        return torch.cat(
            [
                mean_features,
                global_add_pool(node_h, batch),
                global_max_pool(node_h, batch),
            ],
            dim=-1,
        )

    def forward(self, data, extra_features):
        x = data.x                    # [N, node_feat_dim]
        edge_attr = data.edge_attr    # [E, edge_feat_dim]
        edge_index = data.edge_index  # [2, E]
        batch = data.batch            # [N]
        batch_size = data.num_graphs
        num_nodes = x.size(0)

        if x.size(-1) != self.node_feat_dim:
            raise ValueError(
                f"Expected node feature dim {self.node_feat_dim}, but got {x.size(-1)}"
            )
        if edge_attr.size(-1) != self.edge_feat_dim:
            raise ValueError(
                f"Expected edge feature dim {self.edge_feat_dim}, but got {edge_attr.size(-1)}"
            )

        if extra_features is None:
            if self.extra_dim > 0:
                raise ValueError("extra_features cannot be None when extra_dim > 0")
            extra_features = torch.zeros(batch_size, 0, dtype=x.dtype, device=x.device)
        else:
            if extra_features.dim() == 1:
                extra_features = extra_features.unsqueeze(0)
            if extra_features.size(0) != batch_size:
                raise ValueError(
                    f"extra_features batch size {extra_features.size(0)} "
                    f"does not match graph batch size {batch_size}"
                )
            if extra_features.size(-1) != self.extra_dim:
                raise ValueError(
                    f"Expected extra feature dim {self.extra_dim}, "
                    f"but got {extra_features.size(-1)}"
                )
            extra_features = extra_features.to(x.device)

        if edge_index.size(1) == 0:
            node_messages = torch.zeros(
                num_nodes, self.edge_output_dim, dtype=x.dtype, device=x.device
            )
            new_node_feats = torch.cat([x, node_messages], dim=1)
            node_h = self.node_mlp(new_node_feats)

            graph_features = self._pool_graph(node_h, batch)

            combined_features = torch.cat([graph_features, extra_features], dim=-1)
            return self.head(combined_features)

        src, dst = edge_index[0], edge_index[1]

        edge_input = torch.cat([x[src], edge_attr], dim=1)
        h0 = self.edge_mlp(edge_input)   # [E, edge_output_dim]
        h = h0.clone()
        reverse_edge_idx = self.setup_reverse_edges(edge_index, num_nodes)

        # Directed message passing rounds
        for _ in range(self.num_rounds):
            incoming_sum = self._scatter_add(h, dst, num_nodes)  # [N, edge_output_dim]
            directed_message = incoming_sum[src] - h[reverse_edge_idx]
            updated = self.edge_update_mlp(directed_message)
            h = self.activation(updated + h0)
            h = self.dropout(h)

        node_messages = self._scatter_add(h, dst, num_nodes)  # [N, edge_output_dim]
        new_node_feats = torch.cat([x, node_messages], dim=1)
        node_h = self.node_mlp(new_node_feats)  # [N, node_output_dim]

        graph_features = self._pool_graph(node_h, batch)
        combined_features = torch.cat([graph_features, extra_features], dim=-1)

        # Predict
        output = self.head(combined_features)
        return output
