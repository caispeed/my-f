import torch
import torch.nn as nn
import torch.nn.functional as F

ELEMENT_TYPES = [
    "nmos_DG", "nmos_GS", "nmos_DS",
    "pmos_DG", "pmos_GS", "pmos_DS",
    "balun_D1", "balun_D2",
    "resistor", "capacitor", "inductor",
    "vsource", "isource", "port"
]

SCALE_FACTORS = {
    'capacitor': {'c': 1e-12},
    'resistor': {'r': 1e3},
    'inductor': {'l': 1e-9},
    'nmos': {'m': 1.0, 'w': 1e-6},
    'pmos': {'m': 1.0, 'w': 1e-6},
    'vsource': {'dc': 1.0, 'mag': 1.0, 'phase': 1.0},
    'isource': {'dc': 1e-3, 'mag': 1e-3},
    'port': {'dbm': 1.0, 'dc': 1.0, 'freq': 1e9, 'num': 1.0},
    'balun': {'rout': 1.0}
}

element_to_idx = {t: i for i, t in enumerate(ELEMENT_TYPES)}


class EdgeEncoder(nn.Module):
    def __init__(self, out_dim, param_templates, str_params_templates):
        super().__init__()

        SOURCE_TYPES = list(str_params_templates['source_type'].keys())
        self.source_to_idx = {s: i for i, s in enumerate(SOURCE_TYPES)}

        self.type_embed = nn.Embedding(len(ELEMENT_TYPES), out_dim)
        self.source_embed = nn.Embedding(len(SOURCE_TYPES), out_dim)

        # Create one MLP per element type
        self.param_mlps = nn.ModuleDict({
            t: nn.Sequential(
                nn.Linear(len(param_templates[t]), out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            ) for t in param_templates
        })

        self.final = nn.Sequential(
            nn.Linear(3 * out_dim, out_dim),
            nn.ReLU(),
        )

        self.param_templates = param_templates
        self.out_dim = out_dim  # 保存维度供后续使用

    def forward(self, edge_features):
        # 1. 获取当前模型所在的设备 (动态获取，拒绝硬编码 'cpu')
        device = next(self.parameters()).device
        num_edges = len(edge_features)

        # 2. 一次性生成离散特征 IDs 并移动到正确设备
        type_ids = torch.tensor([element_to_idx.get(e['type'], 0) for e in edge_features], dtype=torch.long,
                                device=device)
        source_ids = torch.tensor(
            [self.source_to_idx.get(e['source_type'], 0) if e['source_type'] else 0 for e in edge_features],
            dtype=torch.long, device=device)

        type_emb = self.type_embed(type_ids)
        source_emb = self.source_embed(source_ids)

        # ==========================================================
        #  矩阵批处理 (Batched Vectorization)
        # ==========================================================
        # 预先分配一个致密的显存块，避免海量的小 tensor cat
        param_vecs = torch.zeros((num_edges, self.out_dim), device=device)

        # 3a. 按基类 (base_type) 分组收集边缘索引和参数
        type_groups = {}
        for i, e in enumerate(edge_features):
            base_type = e['type'].split("_")[0]
            if base_type not in type_groups:
                type_groups[base_type] = {'indices': [], 'params': []}

            type_groups[base_type]['indices'].append(i)

            # 确保 params 是 tensor
            p = e['params']
            if not isinstance(p, torch.Tensor):
                p = torch.tensor(p, dtype=torch.float32)
            type_groups[base_type]['params'].append(p)

        # 3b. 对每个分组进行一次性的批处理前向传播 (将几万次计算缩减为十几次矩阵乘法)
        for base_type, group in type_groups.items():
            indices = group['indices']
            # 将同类别的所有边参数堆叠成一个形状为 [N, param_dim] 的大矩阵
            batched_params = torch.stack(group['params']).to(device)

            param_names = self.param_templates[base_type]
            scale_dict = SCALE_FACTORS.get(base_type, {})

            # 创建缩放矩阵并执行向量化除法
            scales = torch.tensor([scale_dict.get(name, 1.0) for name in param_names], dtype=torch.float32,
                                  device=device)
            scaled_params = batched_params / scales

            out = self.param_mlps[base_type](scaled_params)

            # 将计算结果按索引精准填回预分配的大矩阵
            param_vecs[indices] = out
        # ==========================================================

        # 4. 最终拼接 (只需要拼接3个大矩阵，而不是几万个小矩阵)
        concat = torch.cat([type_emb, source_emb, param_vecs], dim=-1)

        return self.final(concat)