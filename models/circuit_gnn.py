import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool

from utils.encoders import EdgeEncoder


class CircuitGNN(MessagePassing):
    def __init__(self, hidden_dim, out_dim, param_templates, str_params_templates, num_layers=4,
                 use_light_norm=True,  # 轻量级归一化（仅增加少量计算，无参数）
                 use_sparse_linear=False):  # 可选：超轻量稀疏线性层（仅1个小矩阵）
        super(CircuitGNN, self).__init__(aggr='add')

        self.num_layers = num_layers
        self.use_light_norm = use_light_norm
        self.use_sparse_linear = use_sparse_linear

        # 1. 节点和边的初始特征编码器 (保持不变)
        self.edge_encoder = EdgeEncoder(hidden_dim, param_templates, str_params_templates)
        self.node_encoder = nn.Linear(in_features=1, out_features=hidden_dim)

        # 【修复1：注册可学习的边权重参数】
        # 仅增加 1 个标量参数，用于平衡节点自身特征与边（元器件）特征的比例
        self.edge_weight = nn.Parameter(torch.ones(1))

        # 【创新点：递归参数共享（Recursive Parameter Sharing）】
        # 相比原 num_layers 个 Linear，仅增加1个参数矩阵，复用于所有层
        if self.use_sparse_linear:
            self.sparse_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
            nn.init.normal_(self.sparse_linear.weight, mean=0.0, std=1e-3)

        # 2. 最终的输出预测层
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        nn.init.normal_(self.output_mlp[3].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.output_mlp[3].bias)

    def forward(self, data):
        x, edge_index, edge_features, batch = data.x, data.edge_index, data.edge_features, data.batch
        device = next(self.parameters()).device

        # =====================================================================
        # 🚨 核心性能修复区 🚨
        # =====================================================================
        # 1. 仅做列表展平，绝不要在这里做 element-wise 的 .to(device) 操作！
        if isinstance(edge_features, list):
            if isinstance(edge_features[0], list):
                edge_features = [ef for sublist in edge_features for ef in sublist]
            # 删除了拖慢速度的逐个 to(device) 循环

        # 2. 让 encoder 在 CPU 上一次性把混合列表编码为一个致密的 Tensor
        if not torch.is_tensor(edge_features):
            edge_attr = self.edge_encoder(edge_features)
        else:
            edge_attr = self.edge_encoder(edge_features.to(device))  # 如果已经是Tensor则正常处理

        # 3. 核心修复：只进行一次性的设备搬运！极大减少 PCIe 通信开销
        edge_attr = edge_attr.to(device)
        x = x.to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device) if batch is not None else batch
        # =====================================================================

        # 编码节点初始特征
        x = self.node_encoder(x)  # [num_nodes, hidden_dim]

        # 【核心轻量化聚合 + 轻量级优化】
        for _ in range(self.num_layers):
            m = self.propagate(edge_index, x=x, edge_attr=edge_attr)

            # 超轻量参数化（共享权重映射）
            if self.use_sparse_linear:
                m = self.sparse_linear(m)

            # 轻量级归一化：无参数的 LayerNorm，防止多层聚合后特征值爆炸（Over-smoothing）
            if self.use_light_norm:
                # 规范化最后一维 (hidden_dim)，不使用 weight 和 bias
                m = F.layer_norm(m, normalized_shape=[m.size(-1)])

            # 残差连接
            x = x + m

        # 全局读出 + 输出预测
        graph_repr = global_mean_pool(x, batch)
        return self.output_mlp(graph_repr)

    def message(self, x_j, edge_attr):
        # 优化1：边特征加权
        # 使用 __init__ 中定义的 self.edge_weight
        msg = x_j + self.edge_weight * edge_attr

        # 优化2：轻量非线性
        msg = F.relu(msg)

        return msg