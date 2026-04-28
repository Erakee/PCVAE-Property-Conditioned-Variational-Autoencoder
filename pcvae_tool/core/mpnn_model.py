"""MPNN 模型（焓预测）— 来自 model/mpnn.py，仅保留生产路径。"""
from __future__ import annotations
import torch
from torch.nn import Linear, ReLU, BatchNorm1d, Module, Sequential
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_scatter import scatter


class MPNNLayer(MessagePassing):
    def __init__(self, emb_dim=256, edge_dim=10, aggr='add'):
        super().__init__(aggr=aggr)
        self.emb_dim = emb_dim
        self.edge_dim = edge_dim
        self.mlp_msg = Sequential(
            Linear(2 * emb_dim + edge_dim, emb_dim), BatchNorm1d(emb_dim), ReLU(),
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU(),
        )
        self.mlp_upd = Sequential(
            Linear(2 * emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU(),
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU(),
        )

    def forward(self, h, edge_index, edge_attr):
        return self.propagate(
            edge_index, h=h, edge_attr=edge_attr,
            size=(h.size(0), h.size(0)))

    def message(self, h_i, h_j, edge_attr):
        return self.mlp_msg(torch.cat([h_i, h_j, edge_attr], dim=-1))

    def aggregate(self, inputs, index, dim_size=None):
        return scatter(inputs, index, dim=self.node_dim,
                       dim_size=dim_size, reduce=self.aggr)

    def update(self, aggr_out, h):
        if aggr_out.size(0) < h.size(0):
            zeros = torch.zeros(h.size(0) - aggr_out.size(0),
                                aggr_out.size(1), device=h.device)
            aggr_out = torch.cat([aggr_out, zeros], dim=0)
        return self.mlp_upd(torch.cat([h, aggr_out], dim=-1))


class MPNNModel(Module):
    def __init__(self, num_layers=2, emb_dim=256, in_dim=42, edge_dim=10, out_dim=1):
        super().__init__()
        self.lin_in = Linear(in_dim, emb_dim)
        self.convs = torch.nn.ModuleList(
            [MPNNLayer(emb_dim, edge_dim, aggr='add') for _ in range(num_layers)])
        self.pool = global_mean_pool
        self.lin_pred = Linear(emb_dim, out_dim)

    def forward(self, data):
        h = self.lin_in(data.x)
        for conv in self.convs:
            h = h + conv(h, data.edge_index, data.edge_attr)
        h_graph = self.pool(h, data.batch)
        return self.lin_pred(h_graph).view(-1)
