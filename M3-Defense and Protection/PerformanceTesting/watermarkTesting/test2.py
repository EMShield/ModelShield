# 第一步：计算0层O层的B*A矩阵
import torch
 
def calculate_O_matrix(lora_path):
    # 加载LoRA权重文件
    lora_data = torch.load(lora_path, map_location=torch.device('cpu'))
    
    # 获取A矩阵和B矩阵
    lora_A = lora_data['model.layers.0.self_attn.o_proj.lora_A.default.weight']
    lora_B = lora_data['model.layers.0.self_attn.o_proj.lora_B.default.weight']
    
    # 计算O矩阵 (B*A)
    O_matrix = torch.matmul(lora_B, lora_A)
    
    return O_matrix, lora_A, lora_B

lora_weights_path = "./lora_weights.pt"  # 替换为实际的LoRA权重文件路径
O_matrix, lora_A, lora_B = calculate_O_matrix(lora_weights_path)


# 第二步：uuid映射O矩阵
import numpy as np

def split_O_matrix(O_matrix):
    """将2048*2048的O矩阵分成四个1024*1024的子矩阵"""
    size = O_matrix.shape[0] // 2
    O1 = O_matrix[:size, :size]
    O2 = O_matrix[:size, size:]
    O3 = O_matrix[size:, :size]
    O4 = O_matrix[size:, size:]
    return O1, O2, O3, O4

def uuid_to_binary(uuid_str):
    """将UUID字符串转换为128位二进制，并分成4个32位"""
    # 移除所有破折号并转换为二进制
    uuid_int = int(uuid_str, 16)
    uuid_bin = format(uuid_int, '0128b')
    # 分成4个32位
    return [uuid_bin[i:i+32] for i in range(0, 128, 32)]

def adjust_indices(indices):
    """调整重复的索引值"""
    seen = set()
    result = []
    for idx in indices:
        while idx in seen:
            idx += 1
        seen.add(idx)
        result.append(idx)
    return result

def extract_vectors(O_part, uuid_part):
    """从O矩阵的一个部分提取三个向量"""
    # 提取前30位，分成3个10位
    indices = [int(uuid_part[i:i+10], 2) - 1 for i in range(0, 30, 10)]
    # 处理重复索引
    indices = adjust_indices(indices)

    # 获取后两位作为提取方式的标志
    extract_mode = int(uuid_part[30:], 2)  # 将最后两位转为整数(0-3)

    vectors = []
    for idx in indices:
        if extract_mode == 0:  # 按行取
            vector = O_part[idx % 1024, :].reshape(1, -1)
        elif extract_mode == 1:  # 按列取
            vector = O_part[:, idx % 1024].reshape(1, -1)
        elif extract_mode == 2:  # 按正对角线取
            # 从idx位置开始的对角线元素
            diag_vector = []
            for i in range(1024):
                row = (idx + i) % 1024
                col = (idx + i) % 1024
                diag_vector.append(O_part[row, col])
            vector = torch.tensor(diag_vector).reshape(1, -1)
        else:  # 按反对角线取
            # 从idx位置开始的反对角线元素
            anti_diag_vector = []
            for i in range(1024):
                row = (idx + i) % 1024
                col = (1023 - ((idx + i) % 1024))
                anti_diag_vector.append(O_part[row, col])
            vector = torch.tensor(anti_diag_vector).reshape(1, -1)

        vectors.append(vector)

    return vectors

# 其余代码保持不变
def process_matrix(O_matrix, uuid):
    # 1. 分割O矩阵
    O1, O2, O3, O4 = split_O_matrix(O_matrix)

    # 2. 将UUID转换为二进制并分割
    uuid_parts = uuid_to_binary(uuid)

    # 3. 从每个O矩阵部分提取向量
    O_parts = [O1, O2, O3, O4]
    all_vectors = []

    for O_part, uuid_part in zip(O_parts, uuid_parts):
        vectors = extract_vectors(O_part, uuid_part)
        all_vectors.extend(vectors)

    # 4. 重组向量为x, y, z
    x = torch.cat([all_vectors[0], all_vectors[3], all_vectors[6], all_vectors[9]], dim=0)
    y = torch.cat([all_vectors[1], all_vectors[4], all_vectors[7], all_vectors[10]], dim=0)
    z = torch.cat([all_vectors[2], all_vectors[5], all_vectors[8], all_vectors[11]], dim=0)

    return x, y, z, all_vectors

# 使用示例
uuid = "c1d0f855d0f841d00c1d0000000000c1"
x, y, z, all_vectors = process_matrix(O_matrix, uuid)

combined = combined_matrix = torch.cat([x, y, z], dim=0)
print("嵌入水印矩阵的维度：", combined.shape)


def reverse_mapping(O_matrix, uuid, C_new):
    """
    将扰动后的C矩阵(12*1024)中的向量放回原始位置

    参数:
    - O_matrix: 原始的2048*2048矩阵
    - uuid: 用于确定原始提取位置的uuid
    - C_new: 扰动后的12*1024矩阵，包含12个行向量

    返回:
    - 四个新的1024*1024矩阵 O1_new, O2_new, O3_new, O4_new
    """
    # 1. 分割原始O矩阵获取形状参考
    O1, O2, O3, O4 = split_O_matrix(O_matrix)
    O_parts = [O1.clone(), O2.clone(), O3.clone(), O4.clone()]  # 创建副本

    # 2. 解析uuid获取每个部分的提取模式和位置
    uuid_parts = uuid_to_binary(uuid)

    # 3. 从C_new中提取向量
    # x: 0-3行, y: 4-7行, z: 8-11行
    x_vectors = C_new[0:4]   # 4*1024
    y_vectors = C_new[4:8]   # 4*1024
    z_vectors = C_new[8:12]  # 4*1024

    # 4. 对每个O部分进行处理
    for part_idx, (O_part, uuid_part) in enumerate(zip(O_parts, uuid_parts)):
        # 获取该部分的三个索引
        indices = [int(uuid_part[i:i+10], 2) - 1 for i in range(0, 30, 10)]
        indices = adjust_indices(indices)  # 处理重复索引

        # 获取提取模式i
        extract_mode = int(uuid_part[30:], 2)

        # 获取对应的新向量
        x_vec = x_vectors[part_idx].view(1, -1)
        y_vec = y_vectors[part_idx].view(1, -1)
        z_vec = z_vectors[part_idx].view(1, -1)
        vectors = [x_vec, y_vec, z_vec]

        # 将向量放回对应位置
        for idx, vec in zip(indices, vectors):
            idx = idx % 1024  # 确保索引在范围内
            if extract_mode == 0:  # 按行取
                O_parts[part_idx][idx, :] = vec.squeeze()

            elif extract_mode == 1:  # 按列取
                O_parts[part_idx][:, idx] = vec.squeeze()

            elif extract_mode == 2:  # 按正对角线取
                for i in range(1024):
                    row = (idx + i) % 1024
                    col = (idx + i) % 1024
                    O_parts[part_idx][row, col] = vec[0, i]

            else:  # 按反对角线取
                for i in range(1024):
                    row = (idx + i) % 1024
                    col = (1023 - ((idx + i) % 1024))
                    O_parts[part_idx][row, col] = vec[0, i]
    return O_parts[0], O_parts[1], O_parts[2], O_parts[3]

O1_new, O2_new, O3_new, O4_new = reverse_mapping(O_matrix, uuid, combined)
print("逆向重组与映射之后的矩阵维度：[2048, 2048]")
