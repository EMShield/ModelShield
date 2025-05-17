import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch.multiprocessing as mp
import logging
from matrix_handler import MatrixHandler

class ComplexMatrixSolver(nn.Module):
    def __init__(self):
        super(ComplexMatrixSolver, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(2048 * 16, 4096),
            nn.LayerNorm(4096),
            nn.LeakyReLU(0.2),
            nn.Linear(4096, 2048),
            nn.LayerNorm(2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2)
        )
        
        # 解码器X
        self.decoder_x = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.LayerNorm(2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 2048 * 8),
            nn.Tanh()
        )
        
        # 解码器Y
        self.decoder_y = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.LayerNorm(2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 8 * 2048),
            nn.Tanh()
        )

    def forward(self, A, B):
        batch_size = 2
        # 在DDP中，每个GPU上批次的大小是平均分配的
        flat_A = A.t().reshape(2, -1)  # (batch_size, 2048 * 8)
        flat_B = B.reshape(2, -1)  # (batch_size, 2048 * 8)
        combined = torch.cat([flat_A, flat_B], dim=1)  # (batch_size, 2048 * 16)
        encoded = self.encoder(combined)  # 输入维度 (batch_size, 2048 * 16)
        x = self.decoder_x(encoded).view(batch_size, 2048, 8)
        y = self.decoder_y(encoded).view(batch_size, 8, 2048)
        
        return x, y

def custom_loss(x, y, A, B, C, alpha=1.0, beta=0.01):
    A = A.view(2, 8, 2048)  # 将 A 调整为 [2, 8, 2048]
    B = B.view(2, 2048, 8)  # 将 B 调整为 [2, 2048, 8]
    C = C.view(2, 2048, 2048)  # 将 C 调整为 [2, 2048, 2048]
    BA = torch.bmm(B, A)
    By = torch.bmm(B, y)
    xA = torch.bmm(x, A)
    xy = torch.bmm(x, y)
    
    prediction = By + xA + xy + BA
    
    # MSE损失
    main_loss = torch.mean(torch.abs(prediction - C))
    
    # L1正则化
    reg_loss = torch.mean(torch.abs(x)) + torch.mean(torch.abs(y))
    
    return main_loss + beta * reg_loss

def generate_batch_data(batch_size, device):
    lora_weights_path = "./lora_weights.pt"  # 替换为实际的LoRA权重文件路径
    matrix_handler = MatrixHandler(lora_weights_path)
    idx = np.random.randint(0, 16)
    idx1 = np.random.randint(0, 16)
    # 获取矩阵
    A, B, C = matrix_handler.get_matrices(idx)
    A1, B1, C1 = matrix_handler.get_matrices(idx1)
    # 将矩阵转换为所需的形状
    A = torch.cat((A, A1), dim=0)  # 在行上堆积，得到 [8, 4096]
    B = torch.cat((B, B1), dim=0)  # 在行上堆积，得到 [4096, 8]
    C = torch.cat((C, C1), dim=0)  # 在行上堆积，得到 [4096, 2048]
    
    # 将矩阵移动到指定设备
    A, B, C = A.to(device), B.to(device), C.to(device)
    
    return A, B, C

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """
    清理分布式训练
    """
    dist.destroy_process_group()

def save_checkpoint(model, optimizer, epoch, loss, filename):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filename)

def train_process(rank, world_size):
    # 设置日志
    logging.basicConfig(
        filename=f'logs/training_{rank}.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    setup(rank, world_size)
    
    if rank == 0:
        logging.info(f"Starting training on {world_size} GPUs")
    
    # 创建模型
    model = ComplexMatrixSolver().to(rank)
    model = DDP(model, device_ids=[rank])
    
    # 优化器设置
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=2000, T_mult=2, eta_min=1e-6
    )
    
    # 训练参数
    n_epochs = 200000
    batch_size = 32
    losses = []
    best_loss = float('inf')
    # 创建进度条
    pbar = tqdm(range(n_epochs), desc="Training Progress", ncols=100)
    try:
        for epoch in tqdm(range(n_epochs), disable=rank != 0):
            A, B, C = generate_batch_data(batch_size, rank)
            
            optimizer.zero_grad()
            x, y = model(A, B)
            loss = custom_loss(x, y, A, B, C)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            if rank == 0:
                losses.append(loss.item())
                # 更新进度条
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    save_checkpoint(model, optimizer, epoch, loss.item(),
                                 './models/best_model.pth')
                
                if (epoch + 1) % 1000 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    logging.info(f'Epoch [{epoch+1}/{n_epochs}], '
                               f'Loss: {loss.item():.4f}, '
                               f'LR: {current_lr:.6f}')
                    save_checkpoint(model, optimizer, epoch, loss.item(),
                                 f'./models/checkpoint_{epoch+1}.pth')
                
                if (epoch + 1) % 5000 == 0:
                    torch.cuda.empty_cache()
    
    except KeyboardInterrupt:
        if rank == 0:
            logging.info("Training interrupted. Saving current state...")
            save_checkpoint(model, optimizer, epoch, loss.item(),
                        './models/interrupted_model.pth')
    finally:
        if rank == 0:
            save_checkpoint(model, optimizer, n_epochs, loss.item(),
                        './models/final_model.pth')

            # 保存训练曲线
            plt.figure(figsize=(10, 5))
            plt.plot(losses)
            plt.title('Training Loss Over Time')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.savefig('./models/loss_curve.png')
            plt.close()
    
            np.save('./models/losses.npy', np.array(losses))
            logging.info("Training completed")
        
        cleanup()

def main():
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU installation.")

    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 获取可用的GPU数量
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError(f"Requires at least 2 GPUs to run, but got {world_size}")
    
    # 创建必要的目录
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    
    # 启动多进程训练
    mp.spawn(train_process,
            args=(world_size,),
            nprocs=world_size,
            join=True)

if __name__ == "__main__":
    main()