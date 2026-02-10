import streamlit as st
import networkx as nx
import yaml
import os
import torch
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.data import Batch

# --- 1. 解决路径问题 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from data_modules.netlist2graph import netlist_to_graph
from data_modules.graph_convertor import networkx_to_pyg
from utils.visual_utils import plot_netlist_graph
from utils.model_utils import load_model, load_data


CIRCUIT_METRIC_MAP = {
    "LNA (低噪声放大器)": [
        "VoltageGain", "NoiseFigure", "S11", "S22", "DCPowerConsumption"
    ],
    "PA (功率放大器)": [
        "OutputPower", "PAE", "PowerGain", "DrainEfficiency", "PSAT"
    ],
    "Mixer (混频器)": [
        "ConversionGain", "NoiseFigure", "S11", "DCPowerConsumption"
    ],
    "VCO (压控振荡器)": [
        "OscillationFrequency", "PhaseNoise", "TuningRange", "OutputPower", "DCPowerConsumption"
    ],
    "VA (可变增益放大器)": [
        "VoltageGain", "Bandwidth", "VoltageSwing", "DCPowerConsumption"
    ],
    "显示所有 (Debug Mode)": [] # 空列表代表不过滤
}
# --- 2. 页面配置 ---
st.set_page_config(page_title="FALCON: 模拟电路自动化设计平台", layout="wide", page_icon="🦅")
st.title("🦅 FALCON: 模拟电路自动化设计平台")


# --- 3. 缓存加载函数 ---
@st.cache_data
def load_visual_config():
    config_path = os.path.join(current_dir, "config", "visual_config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


@st.cache_data
def load_performance_config():
    config_path = os.path.join(current_dir, "config", "data_config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


@st.cache_data
def load_param_templates():
    """加载参数模板 (定义了每个组件需要提取哪些参数)"""
    config_path = os.path.join(current_dir, "dataset", "param_templates.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


@st.cache_resource
def load_gnn_system():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = load_model(device)
    except Exception as e:
        return None, None, f"模型加载失败: {e}"

    try:
        # 加载 Scalers
        _, scalers, _ = load_data(loader=False, heldout=False)
    except Exception as e:
        return None, None, f"数据 Scalers 加载失败 (请先运行 save_gnn_data.py): {e}"

    return model, scalers, None


# --- 4. 核心修复: 特征提取函数 ---
def extract_edge_features(nx_graph, param_templates):
    """
    手动提取边特征，这是模型 forward 所必须的。
    它将网表中的数值属性转换为模型能理解的 Tensor 列表。
    """
    features_list = []

    # 必须使用 keys=True 保证顺序与 edge_index 一致
    for u, v, key, data in nx_graph.edges(keys=True, data=True):
        comp_type_full = data.get('component', 'unknown')  # e.g. nmos_DG
        base_type = comp_type_full.split('_')[0]  # e.g. nmos

        # 1. 准备数值参数 (Params)
        # 获取该组件需要的参数名列表 (如 ['m', 'w'])
        param_names = param_templates.get(base_type, [])
        numeric_attrs = data.get('numeric_attrs', {})

        param_values = []
        for p_name in param_names:
            # 尝试获取数值，如果缺失则填 0
            val = numeric_attrs.get(p_name, 0.0)
            try:
                val = float(val)
            except:
                val = 0.0
            param_values.append(val)

        # 2. 准备源类型 (Source Type)
        # 某些组件(port, vsource)有 type 属性 (如 'dc', 'sine')
        source_type = numeric_attrs.get('type', 'none')  # 默认为 none

        # 构造特征字典
        features_list.append({
            'type': comp_type_full,
            'params': torch.tensor(param_values, dtype=torch.float32),
            'source_type': str(source_type)
        })

    return features_list


# 加载配置
visual_config = load_visual_config()
edge_colors = visual_config.get("element_colors_paper", {})
data_config = load_performance_config()
perf_metrics = list(data_config.get("Performance", {}).keys())
param_templates = load_param_templates()  # 加载参数模板

# --- 5. 侧边栏 ---
st.sidebar.header("功能导航")
app_mode = st.sidebar.selectbox("选择模式", ["电路拓扑可视化", "性能预测 (演示版)"])
st.sidebar.markdown("---")
st.sidebar.info("💡 提示: 请在终端使用 `streamlit run app.py` 启动")

# ==========================================================
# 模式 1: 电路拓扑可视化
# ==========================================================
if app_mode == "电路拓扑可视化":
    st.header("📂 电路网表可视化")
    uploaded_file = st.file_uploader("上传 Spectre/Spice 网表文件", type=None)

    if uploaded_file is not None:
        content = uploaded_file.getvalue().decode("utf-8", errors="ignore").splitlines()
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.subheader("📝 原始网表")
            st.code("\n".join(content), language="spice")
        with col2:
            st.subheader("🕸️ 电路拓扑图")
            try:
                G = netlist_to_graph(content, {})
                fig = plt.figure(figsize=(8, 8))
                pos = nx.spring_layout(G, seed=42, iterations=100)
                nx.draw_networkx_nodes(G, pos, node_size=300, node_color='black')
                nx.draw_networkx_labels(G, pos, font_color='white', font_size=8)

                legend_handles = []
                added_labels = set()
                for u, v, key, data in G.edges(keys=True, data=True):
                    comp_type = data.get('component', 'unknown').split('_')[0]
                    color = edge_colors.get(comp_type, 'gray')
                    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=color, width=2, alpha=0.8)
                    if comp_type not in added_labels:
                        legend_handles.append(plt.Line2D([0], [0], color=color, lw=2, label=comp_type))
                        added_labels.add(comp_type)

                plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1))
                plt.axis("off")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"解析或绘图失败: {e}")
    st.subheader("交互式探索 (Interactive View)")
    from streamlit_agraph import agraph, Node, Edge, Config

    if uploaded_file is not None and 'G' in locals():
        nodes = []
        edges = []

        for node in G.nodes():
            nodes.append(Node(id=node, label=node, size=15, color="black"))

        for u, v, data in G.edges(data=True):
            comp_type = data.get('component', '?')
            color = edge_colors.get(comp_type.split('_')[0], '#999999')
            edges.append(Edge(source=u, target=v, label=data.get('name', ''), color=color))

        config = Config(width=700, height=500, directed=False, nodeHighlightBehavior=True, highlightColor="#F7A7A6")
        agraph(nodes=nodes, edges=edges, config=config)

# ==========================================================
# 模式 2: 性能预测 (演示版)
# ==========================================================
elif app_mode == "性能预测 (演示版)":
    st.header("🚀 电路性能预测 (GNN Inference)")

    with st.spinner("正在加载 FALCON 模型和配置..."):
        model, scalers, err_msg = load_gnn_system()

    if err_msg:
        st.error(err_msg)
    else:
        st.success("模型加载成功！")

        # --- 布局改进 ---
        st.subheader("1. 配置与上传")
        col_input1, col_input2, col_input3 = st.columns([1, 1, 1])

        with col_input1:
            # 新增：电路家族选择器
            circuit_family = st.selectbox(
                "选择电路家族 (Circuit Family)",
                list(CIRCUIT_METRIC_MAP.keys()),
                index=0,
                help="选择电路类型以过滤无关的性能指标"
            )

        with col_input2:
            netlist_file = st.file_uploader("上传网表 (Netlist)", type=None, key="netlist_pred")

        with col_input3:
            values_file = st.file_uploader("上传参数文件 (Values.yaml)", type=["yaml", "yml"], key="values_pred")

        st.markdown("---")

        if netlist_file is not None:
            values_dict = {}
            if values_file is not None:
                try:
                    values_dict = yaml.safe_load(values_file)
                except:
                    st.error("参数文件格式错误")

            if st.button("开始预测 (Predict)", type="primary"):
                try:
                    # A. 预处理
                    content = netlist_file.getvalue().decode("utf-8", errors="ignore").splitlines()
                    netlist_graph = netlist_to_graph(content, values_dict)
                    pyg_data, _, _ = networkx_to_pyg(netlist_graph)

                    # B. 注入特征
                    edge_features = extract_edge_features(netlist_graph, param_templates)
                    pyg_data.edge_features = edge_features

                    # C. 推理
                    batch = Batch.from_data_list([pyg_data])
                    batch = batch.to(next(model.parameters()).device)

                    model.eval()
                    with torch.no_grad():
                        out = model(batch)

                    # D. 反归一化
                    y_pred_norm = out.cpu().numpy().flatten()
                    y_pred_real = []

                    if len(y_pred_norm) == len(scalers):
                        for i, val in enumerate(y_pred_norm):
                            real_val = scalers[i].inverse_transform([[val]])[0][0]
                            y_pred_real.append(real_val)
                    else:
                        y_pred_real = y_pred_norm

                    # E. 智能展示结果 (Filtering)
                    st.subheader(f"📊 预测结果: {circuit_family.split(' ')[0]}")

                    # 获取当前选择电路关心的指标列表
                    target_metrics = CIRCUIT_METRIC_MAP[circuit_family]

                    results = []
                    for i, metric_name in enumerate(perf_metrics):
                        if i < len(y_pred_real):
                            val = y_pred_real[i]

                            # 核心过滤逻辑：
                            # 1. 如果选择了具体家族，只显示该家族列表里的指标
                            # 2. 如果是 "显示所有"，则显示所有非零值
                            is_relevant = (len(target_metrics) == 0) or (metric_name in target_metrics)

                            if is_relevant:
                                results.append({
                                    "指标 (Metric)": metric_name,
                                    "预测值": f"{val:.4f}",
                                    "单位": "dB/dBm/Hz..."  # 这里可以进一步细化单位
                                })

                    if not results:
                        st.warning("所有相关指标的预测值均为 0，请检查输入文件是否正确。")
                    else:
                        # 使用 columns 展示让表格不那么宽
                        st.dataframe(pd.DataFrame(results), use_container_width=False, width=600)

                except Exception as e:
                    st.error(f"预测错误: {e}")
                    import traceback

                    st.text(traceback.format_exc())