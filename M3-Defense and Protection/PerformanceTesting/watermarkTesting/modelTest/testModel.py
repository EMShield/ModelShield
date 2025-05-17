## 然后再以O'作为模型输入，生成x,y

import torch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch.multiprocessing as mp  # 添加这行
import logging  # 添加这行

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
        batch_size = A.size(0)
        flat_A = A.view(batch_size, -1)
        flat_B = B.view(batch_size, -1)
        combined = torch.cat([flat_A, flat_B], dim=1)
        
        encoded = self.encoder(combined)
        x = self.decoder_x(encoded).view(batch_size, 2048, 8)
        y = self.decoder_y(encoded).view(batch_size, 8, 2048)
        
        return x, y

def load_model(model_path):
    """
    加载预训练模型
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ComplexMatrixSolver().to(device)
    
    # 加载模型权重
    state_dict = torch.load(model_path, map_location=device)
    # 移除"module."前缀（如果有的话）
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    
    model.eval()
    return model, device

def calculate_error(x, y, A, B, C):
    """
    计算误差矩阵和相对误差
    """
    # 计算预测
    BA = torch.bmm(B, A)
    By = torch.bmm(B, y)
    xA = torch.bmm(x, A)
    xy = torch.bmm(x, y)
    
    # 计算误差矩阵
    prediction = BA + By + xA + xy
    error_matrix = C - prediction
    
    # 计算相对误差
    relative_error = torch.norm(error_matrix) / torch.norm(C)
    
    return error_matrix, relative_error.item()

def infer(A, B, C, model_path='./model/checkpoint_2000.pth'):
    """
    使用预训练模型进行推理
    """
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ComplexMatrixSolver().to(device)
    
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 确保输入是批次形式
    if len(A.shape) == 2:
        A = A.unsqueeze(0)
        B = B.unsqueeze(0)
        C = C.unsqueeze(0)
    
    # 移动数据到GPU
    A = A.to(device)
    B = B.to(device)
    C = C.to(device)
    
    # 推理
    with torch.no_grad():
        x, y = model(A, B)
        error_matrix, relative_error = calculate_error(x, y, A, B, C)
    
    # 如果输入不是批次，则移除批次维度
    if len(A.shape) == 3 and A.shape[0] == 1:
        x = x.squeeze(0)
        y = y.squeeze(0)
        error_matrix = error_matrix.squeeze(0)
    
    return x.cpu(), y.cpu(), error_matrix.cpu(), relative_error

def combine_O_matrix(O1, O2, O3, O4):
    """
    将四个1024*1024的矩阵重组为一个2048*2048的矩阵
    
    参数:
    - O1, O2, O3, O4: 四个1024*1024的矩阵
    
    返回:
    - O_new: 2048*2048的矩阵
    """
    # 先水平拼接
    top_half = torch.cat([O1, O2], dim=1)     # 1024 * 2048
    bottom_half = torch.cat([O3, O4], dim=1)   # 1024 * 2048
    
    # 再垂直拼接
    O_new = torch.cat([top_half, bottom_half], dim=0)  # 2048 * 2048
    
    return O_new

if __name__ == '__main__':
    # 假设 lora_A 和 lora_B 已经定义并计算

    # 动态读取 lora_A, lora_B 和 O_new
    lora_A_loaded = torch.load('lora_A.pt')
    lora_B_loaded = torch.load('lora_B.pt')
    O_new_loaded = torch.load('O_new.pt')

    # 使用加载的矩阵进行推理
    x, y, error_matrix, relative_error = infer(lora_A_loaded, lora_B_loaded, O_new_loaded)
    
    print(f"输入形状:")
    print(f"A shape: {lora_A_loaded.shape}")
    print(f"B shape: {lora_B_loaded.shape}")
    print(f"C shape: {O_new_loaded.shape}")
    
    print(f"\n输出形状:")
    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")
    print(f"Error matrix shape: {error_matrix.shape}")
    print(f"Relative error: {relative_error:.6f}")
