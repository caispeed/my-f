import torch
import torch.nn as nn
import torch.nn.functional as F

# ELEMENT_TYPES = [
#     "nmos", "pmos", "resistor", "capacitor", "vsource",
#     "port", "inductor", "balun", "isource"
# ]
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

        # REGION_TYPES = list(str_params_templates['region'].keys())
        SOURCE_TYPES = list(str_params_templates['source_type'].keys())
        # self.region_to_idx = {r: i for i, r in enumerate(REGION_TYPES)}
        self.source_to_idx = {s: i for i, s in enumerate(SOURCE_TYPES)}

        self.type_embed = nn.Embedding(len(ELEMENT_TYPES), out_dim)
        # self.region_embed = nn.Embedding(len(REGION_TYPES), out_dim)
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

    def forward(self, edge_features):
        # 获取模型所在的设备
        device = next(self.parameters()).device

        # 确保 type_ids 和 source_ids 在正确的设备上
        type_ids = torch.tensor([element_to_idx.get(e['type'], 0) for e in edge_features], device=device)
        source_ids = torch.tensor(
            [self.source_to_idx.get(e['source_type'], 0) if e['source_type'] else 0 for e in edge_features],
            device=device)

        type_emb = self.type_embed(type_ids)
        source_emb = self.source_embed(source_ids)

        param_vecs = []
        for e in edge_features:
            comp_type = e['type']  # e.g., 'nmos_DG'
            base_type = comp_type.split("_")[0]
            params = e['params']  # already a tensor

            # 确保 params 在正确的设备上
            if torch.is_tensor(params):
                params = params.to(device)

            param_names = self.param_templates[base_type]
            scale_dict = SCALE_FACTORS.get(base_type, {})

            # Scale each parameter individually
            scaled_values = [
                params[i] / scale_dict.get(param_names[i], 1.0)
                for i in range(len(param_names))
            ]
            # 确保 scaled_tensor 在正确的设备上
            scaled_tensor = torch.stack(scaled_values).to(device)

            mlp = self.param_mlps[base_type]
            param_vecs.append(mlp(scaled_tensor.unsqueeze(0)))

        param_vecs = torch.cat(param_vecs, dim=0)  # [num_edges, out_dim]
        concat = torch.cat([type_emb, source_emb, param_vecs], dim=-1)

        return self.final(concat)
