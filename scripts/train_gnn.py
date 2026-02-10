import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.circuit_gnn import CircuitGNN
from utils.io_tools import load_yaml, seed_everything
from train.train_eval_gnn import train, evaluate
from utils.visual_utils import plot_loss_curves
from utils.model_utils import init_weights


def main():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    root_dir = "./dataset"
    data_config = load_yaml('./config/data_config.yaml')
    global_perf_list = list(data_config['Performance'].keys())
    circuit_to_code = data_config['Classes']

    # ==== 优化参数 ====
    batch_size = 512
    num_epochs = 200
    patience = 10
    lr = 1e-3
    hidden_dim = 128

    # ==== 数据加载优化 ====
    gnn_data = torch.load(f"{root_dir}/gnn_data.pt", weights_only=False)

    # 使用数据子集进行快速验证
    train_subset_size = min(20000, len(gnn_data["train"]))
    val_subset_size = min(2000, len(gnn_data["val"]))

    train_data = Subset(gnn_data["train"], range(train_subset_size))
    val_data = Subset(gnn_data["val"], range(val_subset_size))

    print(f"[INFO] Using {len(train_data)} train samples, {len(val_data)} validation samples for fast training.")

    # Windows上禁用多进程
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                             num_workers=0)  # 设置为0
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                           num_workers=0)    # 设置为0

    param_templates = load_yaml('./Dataset/param_templates.yaml')
    str_params_templates = load_yaml('./Dataset/str_params_templates.yaml')

    # ==== 模型优化 ====
    model = CircuitGNN(
        hidden_dim=hidden_dim,
        out_dim=len(global_perf_list),
        param_templates=param_templates,
        str_params_templates=str_params_templates
    ).to(device)

    model.apply(init_weights)

    # ==== 优化器优化 ====
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    # ==== 训练循环优化 ====
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []

    print("[INFO] Starting optimized training...")

    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 每10个epoch打印一次进度
        if epoch % 10 == 0 or epoch == 1:
            print(f"🎯 Epoch {epoch}/{num_epochs}")
            print(f"  🟢 Train Loss: {train_loss:.4f}")
            print(f"  🔵 Val Loss: {val_loss:.4f}")
            print(f"  📊 Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "./checkpoints/best_gnn_model.pt")
            if epoch % 10 == 0:
                print("  💾 Saved best model.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  ⏹️ Early stopping triggered at epoch {epoch}.")
                break

    # 确保目录存在
    os.makedirs("./plots/gnn", exist_ok=True)
    plot_loss_curves(train_losses, val_losses, title="GNN Training Loss",
                    save_path='./plots/gnn/GNNLoss.pdf', log_scale=False)

    print(f"[INFO] Training completed. Best validation loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    main()