# 第一步：计算0层O层的B*A矩阵
import torch

def calculate_O_matrix(lora_path):
    # 加载LoRA权重文件
    lora_data = torch.load(lora_path, map_location=torch.device('cpu'))
    
    # 获取A矩阵和B矩阵
    lora_A = lora_data['model.layers.0.self_attn.o_proj.lora_A.default.weight']
    lora_B = lora_data['model.layers.0.self_attn.o_proj.lora_B.default.weight']
    
    O_matrix = torch.matmul(lora_B, lora_A)
    
    return O_matrix

lora_weights_path = "./lora_weights.pt"  # 替换为实际的LoRA权重文件路径
O_matrix = calculate_O_matrix(lora_weights_path)

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

def extract_vectors(O_part, uuid_part):
    """从O矩阵的一个部分提取三个向量"""
    # 提取前30位，分成3个10位
    indices = [int(uuid_part[i:i+10], 2) - 1 for i in range(0, 30, 10)]
    # 获取第31位作为方向标志
    is_column = bool(int(uuid_part[30]))
    
    vectors = []
    for idx in indices:
        if is_column:
            # 选择列向量并转置为1*1024
            vector = O_part[:, idx].reshape(1, -1)
        else:
            # 选择行向量
            vector = O_part[idx, :].reshape(1, -1)
        vectors.append(vector)
    
    return vectors

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
    
    return x, y, z

# 使用示例
uuid = "c1d0f855d0f841d00c1d0000000000c1"
x, y, z = process_matrix(O_matrix, uuid)

# 第三步：状态机筛选
import random

## 获取状态机初始状态
def get_remaining_bits(uuid_str):
    """从UUID的四个部分中提取最后一位并拼接成4位二进制"""
    # 使用之前的函数将UUID转换为二进制并分段
    uuid_parts = uuid_to_binary(uuid_str)
    
    # 从每个32位部分取最后一位（即第31位，因为索引从0开始）
    remaining_bits = ''.join(part[31] for part in uuid_parts)
    
    return remaining_bits
uuid = "c1d0f855d0f841d00c1d0000000000c1"
four_bits = get_remaining_bits(uuid)
# print(f"剩余的4位二进制: {four_bits}")

## 定义混沌状态机
class NonlinearFSM:
    def __init__(self, secret=None, modulus=16):
        """
        初始化FSM，使用混沌映射的思想
        :param secret: 用于结果映射的列表，长度为 4。默认为 [1, 2, 4, 8]
        :param modulus: 模数，用于限制状态的范围
        """
        self.secret = secret if secret else [1, 2, 4, 8]
        self.modulus = modulus
        # 使用多个无理数作为混沌参数
        self.phi = (1 + 5 ** 0.5) / 2  # 黄金分割比
        self.e = 2.718281828459045     # 自然对数e
        self.pi = 3.141592653589793    # 圆周率π
        self.sqrt2 = 2 ** 0.5          # 根号2
        
    def next_state(self, current_state):
        """
        使用更复杂的混沌映射计算下一个状态
        状态驱动越复杂，越安全，因为这样公布选取了那些行和列，然后
        """
        # 归一化当前状态到[0,1]区间
        x = current_state / self.modulus
        
        # 使用多个混沌映射的组合
        logistic = 4 * x * (1 - x)                    # Logistic映射
        tent = min(2 * x, 2 * (1 - x))               # Tent映射
        sine = abs(np.sin(self.pi * x))              # Sine映射
        cubic = abs((3 * x ** 3 - 2 * x) % 1)        # 三次映射
        
        # 混合多个映射结果
        mixed = (logistic * self.phi + 
                tent * self.e + 
                sine * self.pi + 
                cubic * self.sqrt2) % 1
        
        # 添加非线性扰动
        disturbed = (mixed + x * self.phi) % 1
        
        # 将结果映射回原始范围
        state_prime = int(disturbed * self.modulus)
        
        # 确保不会停留在同一个状态
        if state_prime == current_state:
            state_prime = (state_prime + int(self.phi * 10)) % self.modulus
            
        return state_prime

    def iterate(self, initial_state, iterations):
        """
        根据初始状态和迭代次数生成结果
        """
        current_state = initial_state
        results = []
        for _ in range(iterations):
            current_state = self.next_state(current_state)
            index = current_state % 4
            results.append(self.secret[index])
        return results

## 计算筛选器
fsm = NonlinearFSM(secret=[1, 2, 4, 8])
# 指定初始状态和迭代次数
initial_state = int(four_bits, 2) 
iterations = 1024
results = fsm.iterate(initial_state, iterations)
# print(results)

def convert_to_binary_matrix(numbers, num_rows=4):
    """
    将数字列表转换为二进制矩阵
    numbers: 输入的数字列表 [1,8,4,2,...]
    num_rows: 输出矩阵的行数（默认为4）
    """
    # 创建一个全零矩阵
    num_cols = len(numbers)
    binary_matrix = torch.zeros((num_rows, num_cols))
    
    # 对每个数字进行转换
    for col, num in enumerate(numbers):
        # 根据数字设置对应位置的1
        if num == 1:
            binary_matrix[3, col] = 1  # 0001
        elif num == 2:
            binary_matrix[2, col] = 1  # 0010
        elif num == 4:
            binary_matrix[1, col] = 1  # 0100
        elif num == 8:
            binary_matrix[0, col] = 1  # 1000
            
    return binary_matrix

binary_matrix = convert_to_binary_matrix(results)

## 正式筛选
def filter_matrix(input_matrix, binary_matrix):
    """
    根据binary_matrix对input_matrix进行筛选
    input_matrix: 输入矩阵 (4*1024)
    binary_matrix: 筛选矩阵 (4*1024)，每列只有一个1
    return: 筛选后的向量 (1*1024)
    """
    # 创建结果向量
    result = torch.zeros(1, input_matrix.shape[1])
    
    # 对每一列进行筛选
    for col in range(input_matrix.shape[1]):
        # 找到binary_matrix当前列中1的位置
        row = torch.where(binary_matrix[:, col] == 1)[0].item()
        # 将对应位置的值放入结果向量
        result[0, col] = input_matrix[row, col]
    
    return result

# 对x,y,z分别进行筛选
new_x = filter_matrix(x, binary_matrix)
new_y = filter_matrix(y, binary_matrix)
new_z = filter_matrix(z, binary_matrix)

# 第四步：映射到三维空间 + DBSCAN聚类
import numpy as np
from sklearn.cluster import DBSCAN
import torch

def cluster_and_visualize(x, y, z, scale=10000, eps=1.0, min_samples=5):
    """
    对三维点进行DBSCAN聚类并可视化
    x, y, z: 1*1024的向量
    scale: 坐标放大倍数
    eps: DBSCAN的邻域半径参数
    min_samples: DBSCAN的最小样本数参数
    """
    # 数据预处理：放大坐标
    x_scaled = x.numpy().flatten() * scale
    y_scaled = y.numpy().flatten() * scale
    z_scaled = z.numpy().flatten() * scale

    # 将数据组织成点的形式 (1024, 3)
    points = np.vstack((x_scaled, y_scaled, z_scaled)).T

    # 应用DBSCAN聚类
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_

    # 获取聚类数量（不包括噪声点）
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    print(f'聚类数量: {n_clusters}')

    # 计算每个簇的详细信息
    cluster_info = {}
    for label in unique_labels:
        mask = labels == label
        cluster_points = points[mask]

        # 计算簇的统计信息
        info = {
            'size': len(cluster_points),  # 簇中点的数量
            'center': np.mean(cluster_points, axis=0),  # 簇中心
            'std': np.std(cluster_points, axis=0),  # 标准差
            'min': np.min(cluster_points, axis=0),  # 最小值
            'max': np.max(cluster_points, axis=0),  # 最大值
            'density': len(cluster_points) / (np.prod(np.max(cluster_points, axis=0) - np.min(cluster_points, axis=0)) + 1e-10),  # 密度
            'points': cluster_points  # 保存簇中的所有点
        }

        # 计算点到中心的平均距离
        distances = np.linalg.norm(cluster_points - info['center'], axis=1)
        info['avg_distance_to_center'] = np.mean(distances)
        info['max_distance_to_center'] = np.max(distances)

        cluster_name = 'noise' if label == -1 else f'cluster_{label}'
        cluster_info[cluster_name] = info

        # 打印簇的信息
        print(f"\n{cluster_name.upper()} 信息:")
        print(f"点数: {info['size']}")
        print(f"中心坐标: ({info['center'][0]:.2f}, {info['center'][1]:.2f}, {info['center'][2]:.2f})")
        print(f"标准差: ({info['std'][0]:.2f}, {info['std'][1]:.2f}, {info['std'][2]:.2f})")
        print(f"平均到中心距离: {info['avg_distance_to_center']:.2f}")
        print(f"最大到中心距离: {info['max_distance_to_center']:.2f}")
        print(f"密度: {info['density']:.2e}")

    # 保存详细的聚类结果
    clustering_result = {
        'points': points,
        'labels': labels,
        'n_clusters': n_clusters,
        'cluster_info': cluster_info,
        'params': {
            'scale': scale,
            'eps': eps,
            'min_samples': min_samples
        }
    }
    np.save('clustering_result.npy', clustering_result)

    return clustering_result

# 使用函数
clustering_result = cluster_and_visualize(new_x, new_y, new_z,
                                        scale=10000,  # 坐标放大10000倍
                                        eps=1.0,      # 可以根据放大后的坐标调整
                                        min_samples=5)
