import os, sys, pathlib
import argparse  # ✅ 引入 argparse，用于指定数据集版本

sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import torch
import numpy as np
from copy import copy
import json
from tqdm import tqdm
from torch.utils.data import Subset
from utils.io_tools import load_yaml, seed_everything, convert_list_to_tuple
from utils.visual_utils import plot_relative_error_distribution, plot_loss_backward
from utils.model_utils import load_data, load_model
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.backward_utils import *

with open("./config/clamp_bounds.json", "r") as f:
    CLAMP_BOUNDS_BY_TOPOLOGY = convert_list_to_tuple(json.load(f))

YELLOW = "\033[93m"
RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
WHITE = "\033[97m"
seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def invert_performance_to_params(model, graph_data, idx, param_templates,
                                 results_dict, global_perf_dict, scalers=None,
                                 lr=1e-2, num_steps=500, verbose=False):
    model.eval()
    device = next(model.parameters()).device
    graph_data = copy(graph_data).to(device)

    x_params = initialize_param_vector(CLAMP_BOUNDS_BY_TOPOLOGY, graph_data, device)
    optimizer = torch.optim.Adam([x_params], lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=50)

    best_loss = float("inf")
    best_params = None
    best_param_perf = None
    early_stop_counter = 0
    loss_history = []

    progress_bar = tqdm(range(num_steps),
                        desc=f"Optimizing {graph_data.circuit_type}",
                        leave=False,
                        mininterval=1.0)

    for step in progress_bar:
        optimizer.zero_grad()
        loss, out, _ = run_optimization_step(model, graph_data, x_params, param_templates)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        scheduler.step(loss.item())

        clamp_params(CLAMP_BOUNDS_BY_TOPOLOGY, x_params, graph_data)

        if loss.item() < best_loss - 1e-6:
            best_loss = loss.item()
            best_params = x_params.clone().detach()
            best_param_perf = out.clone().detach()
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= 200:
            break

    best_perf_pred, rel_error, area, success = log_final_info(
        graph_data, best_params, best_param_perf, best_loss, scalers, idx, results_dict, global_perf_dict
    )

    return best_params.detach(), best_perf_pred, best_loss, rel_error, success


def main():
    # =====================================================================
    # 🎯 核心亮点：支持终端传入数据集版本号，与下游 EDA 验证框架无缝对接
    # =====================================================================
    parser = argparse.ArgumentParser(description="FALCON 后推评估与数据集生成")
    parser.add_argument("--version", type=str, default="run_v1_baseline",
                        help="生成的数据集版本文件夹名 (例如: run_v2_finetuned)")
    args = parser.parse_args()

    DATASET_VERSION = args.version
    BASE_DIR = f"./circuits/{DATASET_VERSION}"

    # 初始化记录变量
    relative_errors = []
    best_losses = []
    results_dict = {}
    success_dict = {}
    all_dict = {}
    finished = []

    # 1. 加载数据
    test_dataset, scalers, global_perf_dict = load_data(loader=False, heldout=False)
    model = load_model(device)

    # =====================================================================
    # ✅ 强制提前创建所有电路类型的专属物理文件夹！(基于动态版本号)
    # =====================================================================
    print(f"\n{YELLOW}📂 正在为您预先扫描并创建电路分类文件夹 (数据集版本: {DATASET_VERSION})...{RESET}")
    unique_circuit_types = set([sample.circuit_type for sample in test_dataset])
    for c_type in unique_circuit_types:
        folder_path = f"{BASE_DIR}/{c_type}"
        os.makedirs(folder_path, exist_ok=True)
    print(f"{GREEN}✅ 成功创建 {len(unique_circuit_types)} 个分类文件夹！请去 {BASE_DIR}/ 目录下查看！{RESET}\n")
    # =====================================================================

    print(f"{CYAN}=== Starting Backward Evaluation (Limit: 2000 Samples) ==={RESET}")

    TARGET_COUNT = 2000
    if len(test_dataset) > TARGET_COUNT:
        test_subset = Subset(test_dataset, range(TARGET_COUNT))
    else:
        test_subset = test_dataset

    for i, sample in enumerate(tqdm(test_subset, desc="Total Progress", unit="sample", dynamic_ncols=True)):
        sample = copy(sample)

        if sample.circuit_type in finished:
            continue

        optimized_params, pred_perf, best_loss, rel_error, success = invert_performance_to_params(
            model=model,
            graph_data=sample,
            idx=i,
            global_perf_dict=global_perf_dict,
            results_dict=results_dict,
            scalers=scalers,
            param_templates=load_yaml('./dataset/param_templates.yaml'),
            lr=1e-6,
            num_steps=1000,
            verbose=False
        )

        if success:
            success_dict[sample.circuit_type] = success_dict.get(sample.circuit_type, 0) + 1
            all_dict[sample.circuit_type] = all_dict.get(sample.circuit_type, 0) + 1
            best_losses.append(best_loss)
            relative_errors.append(rel_error)

            df = results_dict.get(sample.circuit_type)
            if df is not None and len(df) == 500 and sample.circuit_type not in finished:
                finished.append(sample.circuit_type)

                # 💡 保存路径动态匹配验证框架
                save_name = f"{BASE_DIR}/{sample.circuit_type}/{sample.circuit_type}_full_500.csv"
                df.to_csv(save_name, index=False)
                print(
                    f"\n{YELLOW}>>> 🎉 Collected 500 successful samples for {sample.circuit_type}! Saved to {save_name}{RESET}")
        else:
            success_dict[sample.circuit_type] = success_dict.get(sample.circuit_type, 0)
            all_dict[sample.circuit_type] = all_dict.get(sample.circuit_type, 0) + 1

    print(f"\n{CYAN}=== Evaluation Finished! Saving all remaining data... ==={RESET}")
    for circuit_type, df in results_dict.items():
        if circuit_type not in finished and df is not None and len(df) > 0:
            # 💡 保存路径动态匹配验证框架
            save_name = f"{BASE_DIR}/{circuit_type}/{circuit_type}_partial_{len(df)}.csv"
            df.to_csv(save_name, index=False)
            print(f"  💾 Saved partial results for {circuit_type}: {len(df)} samples -> {save_name}")

    print(f"\n{GREEN}=== Final Statistics (Success Rate) ==={RESET}")
    for c_type, total in all_dict.items():
        succ = success_dict.get(c_type, 0)
        rate = (succ / total * 100) if total > 0 else 0
        print(f"  • {c_type:<10}: {succ}/{total} ({rate:.2f}%)")


if __name__ == "__main__":
    main()