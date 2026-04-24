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
        device = next(self.parameters()).device

        # 1. 编码元件类型和端口类型
        type_ids = torch.tensor([element_to_idx.get(e['type'], 0) for e in edge_features], device=device)
        source_ids = torch.tensor(
            [self.source_to_idx.get(e['source_type'], 0) if e['source_type'] else 0 for e in edge_features],
            device=device)

        type_emb = self.type_embed(type_ids)
        source_emb = self.source_embed(source_ids)

        param_vecs = []
        # 2. 逐个处理元件参数
        for e in edge_features:
            comp_type = e['type']  # e.g., 'nmos_DG'
            base_type = comp_type.split("_")[0]
            params = e['params']

            if not torch.is_tensor(params):
                params = torch.tensor(params, dtype=torch.float32)
            params = params.to(device)

            param_names = self.param_templates.get(base_type, [])
            scale_dict = SCALE_FACTORS.get(base_type, {})

            # 【核心修复1】动态生成与模型预期参数严格一致的 scales 列表
            scales = torch.tensor(
                [scale_dict.get(p, 1.0) for p in param_names],
                dtype=torch.float32,
                device=device
            )

            # 【核心修复2】暴力防越界对齐：Cadence导出7个，我只要前2个；不够的补1
            if len(params) > len(scales):
                params = params[:len(scales)]  # 截断多余的寄生脏参数
            elif len(params) < len(scales):
                padding = torch.ones(len(scales) - len(params), device=device)
                params = torch.cat([params, padding], dim=0)  # 补齐缺失的参数

            # 现在 7 vs 2 的问题被彻底解决，永远是等长的张量相除
            scaled_params = params / scales

            mlp = self.param_mlps[base_type]
            param_vecs.append(mlp(scaled_params.unsqueeze(0)))

        # 3. 拼接所有特征输出
        if len(param_vecs) > 0:
            param_vecs = torch.cat(param_vecs, dim=0)  # [num_edges, out_dim]
            concat = torch.cat([type_emb, source_emb, param_vecs], dim=-1)
            return self.final(concat)
        else:
            return None