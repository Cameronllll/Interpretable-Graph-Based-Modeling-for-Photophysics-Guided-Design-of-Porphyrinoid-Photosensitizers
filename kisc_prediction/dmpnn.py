import torch
import torch.nn as nn
from enum import Enum
from torch_geometric.nn import global_mean_pool, global_max_pool


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
    def __init__(self, in_feature, hidden_dim, dropout_rate, activation_type):
        super().__init__()
        activation = ActivationFunction[activation_type].value()
        hidden_dim_2 = max(hidden_dim // 2, 32)

        self.net = nn.Sequential(
            nn.LayerNorm(in_feature),
            nn.Linear(in_feature, hidden_dim, bias=True),
            activation,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim_2, bias=True),
            activation,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_2, 1, bias=True),
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
        num_experts,
        moe_hid_dim,
    ):
        super().__init__()

        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.edge_output_dim = edge_output_dim
        self.node_output_dim = node_output_dim
        self.extra_dim = extra_dim
        self.num_rounds = num_rounds

        self.activation = ActivationFunction[activation_type1].value()
        self.dropout = nn.Dropout(dropout_rate)

        self.edge_mlp = nn.Sequential(
            nn.Linear(node_feat_dim + edge_feat_dim, edge_output_dim, bias=False),
            self.activation,
        )

        self.edge_update_mlp = nn.Sequential(
            nn.Linear(edge_output_dim, edge_output_dim, bias=False)
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(node_feat_dim + edge_output_dim, node_output_dim, bias=True),
            self.activation,
            self.dropout,
        )

        pooled_graph_dim = node_output_dim * 2
        self.graph_norm = nn.LayerNorm(pooled_graph_dim)
        self.extra_norm = nn.LayerNorm(extra_dim) if extra_dim > 0 else nn.Identity()

        self.head = RegressionHead(
            in_feature=pooled_graph_dim + extra_dim,
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

    def setup_reverse_edges(self, edge_index: torch.Tensor) -> torch.Tensor:
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        edge_map = {(u, v): i for i, (u, v) in enumerate(zip(src, dst))}
        reverse_idx = [edge_map.get((v, u), i) for i, (u, v) in enumerate(zip(src, dst))]
        return torch.tensor(reverse_idx, dtype=torch.long, device=edge_index.device)

    def _pool_graph(self, node_h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        mean_pool = global_mean_pool(node_h, batch)
        max_pool = global_max_pool(node_h, batch)
        return torch.cat([mean_pool, max_pool], dim=-1)

    def forward(self, data, extra_features):
        x = data.x
        edge_attr = data.edge_attr
        edge_index = data.edge_index
        batch = data.batch
        batch_size = data.num_graphs
        num_nodes = x.size(0)

        if x.size(-1) != self.node_feat_dim:
            raise ValueError(f"Expected node feature dim {self.node_feat_dim}, but got {x.size(-1)}")
        if edge_attr.size(-1) != self.edge_feat_dim:
            raise ValueError(f"Expected edge feature dim {self.edge_feat_dim}, but got {edge_attr.size(-1)}")

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
                    f"Expected extra feature dim {self.extra_dim}, but got {extra_features.size(-1)}"
                )
            extra_features = extra_features.to(x.device)

        if edge_index.size(1) == 0:
            node_messages = torch.zeros(num_nodes, self.edge_output_dim, dtype=x.dtype, device=x.device)
            node_h = self.node_mlp(torch.cat([x, node_messages], dim=1))
            graph_features = self.graph_norm(self._pool_graph(node_h, batch))
            extra_features = self.extra_norm(extra_features)
            return self.head(torch.cat([graph_features, extra_features], dim=-1))
        # 确定边的反向边
        src, dst = edge_index[0], edge_index[1]
        edge_input = torch.cat([x[src], edge_attr], dim=1)
        h0 = self.edge_mlp(edge_input)
        h = h0.clone()
        reverse_edge_idx = self.setup_reverse_edges(edge_index)
        # 进行消息传递
        for _ in range(self.num_rounds):
            incoming_sum = self._scatter_add(h, dst, num_nodes)
            directed_message = incoming_sum[src] - h[reverse_edge_idx]
            updated = self.edge_update_mlp(directed_message)
            h = self.activation(updated + h0)
            h = self.dropout(h)

        node_messages = self._scatter_add(h, dst, num_nodes)
        node_h = self.node_mlp(torch.cat([x, node_messages], dim=1))
        graph_features = self.graph_norm(self._pool_graph(node_h, batch))
        extra_features = self.extra_norm(extra_features)

        output = self.head(torch.cat([graph_features, extra_features], dim=-1))
        return output
