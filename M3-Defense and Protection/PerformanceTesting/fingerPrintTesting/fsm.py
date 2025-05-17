import numpy as np
import torch

def uuid_to_binary(uuid_str):
    """将UUID字符串转换为128位二进制，并分成4个32位"""
    # 移除所有破折号并转换为二进制
    uuid_int = int(uuid_str, 16)
    uuid_bin = format(uuid_int, '0128b')
    # 分成4个32位
    return [uuid_bin[i:i+32] for i in range(0, 128, 32)]

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
print("FSM迭代开始")
results = fsm.iterate(initial_state, iterations)
# print(results)
print("FSM迭代结束")

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
# 打印筛选矩阵形状
print("筛选矩阵形状:", binary_matrix.shape)
# 打印前几列以验证转换是否正确
print("\n筛选矩阵前10列的内容:")
print(binary_matrix[:, :10])
