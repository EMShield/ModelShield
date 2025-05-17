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
    
    
    # 保存O矩阵到文件
    torch.save(O_matrix, 'O_matrix.pt')
    
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



# 第三步：12*1024矩阵转为张量,张量转为初始灰度图像（还有一个张量转回12*1024矩阵的函数）

## 先转为12*1024矩阵
combined = combined_matrix = torch.cat([x, y, z], dim=0)

import torch
import numpy as np

def tensor_to_image(tensor, save_path='./watermark_image.png'):
    """
        将12*1024的矩阵=>归一化=>切割重组=>张量=>另存为48*256的图像
    """
    # 1. 归一化到[0,1]区间
    ## 归一化： 当前值 / 极差
    min_val = tensor.min()
    max_val = tensor.max()
    normalized = (tensor - min_val) / (max_val - min_val)
    
    # 2. 分割成4个12*256的块
    chunks = torch.chunk(normalized, 4, dim=1)  # 在列维度上分成4份
    
    # 3. 垂直堆叠成48*256
    stacked = torch.cat(chunks, dim=0)  # 48*256
    
    return stacked


# tensor是你的12*1024张量
stacked_tensor = tensor_to_image(combined, save_path='./original.png')

def image_to_tensor(stacked_tensor, original_tensor):
    """
    将48*256的stacked_tensor转回12*1024的原始tensor
    （因为嵌入水印是对48*256嵌入的，而且是归一化之后嵌入的，所以这里需要有一个逆归一化，把这种改动放回到B*A中，然后再利用预训练模型将B*A改动放回到B、A中）
    
    参数:
    stacked_tensor: 48*256的张量
    original_tensor: 原始的12*1024张量，用于获取原始值范围
    
    返回:
    restored_tensor: 恢复后的12*1024张量
    """
    # 1. 先将48*256转回12*1024的形状
    chunks = torch.chunk(stacked_tensor, 4, dim=0)  # 在行维度上分成4份，每份12*256
    normalized = torch.cat(chunks, dim=1)  # 水平拼接，变回12*1024
    
    # 2. 逆归一化，恢复到原始值范围
    min_val = original_tensor.min()
    max_val = original_tensor.max()
    restored_tensor = normalized * (max_val - min_val) + min_val
    
    return restored_tensor


# 使用DCT + 扩频序列 + DWT
## 测试一下（现在嵌入水印之后，恢复出来的12*1024矩阵和之前原始矩阵值差多少）



import numpy as np
from scipy.fft import dct, idct
import pywt

class HybridWatermark:
    def __init__(self, block_size=16, alpha=0.15, beta=0.2):
        self.block_size = block_size
        self.alpha = alpha  # DCT水印强度
        self.beta = beta   # DWT水印强度
    def dwt2(self, array):
        """使用DWT替代IWT"""
        return pywt.dwt2(array, 'haar')
    
    def idwt2(self, coeffs):
        """使用IDWT替代IIWT"""
        return pywt.idwt2(coeffs, 'haar')
    
    def generate_watermark(self, length, seed=42):
        np.random.seed(seed)
        return np.random.choice([-1, 1], size=length)
    
    def get_mid_band_positions(self, block_size):
        positions = []
        for i in range(2, 6):
            for j in range(2, 6):
                if 4 <= i + j <= 8:
                    positions.append((i, j))
        return positions
    
    def embed(self, image, watermark_bits):
        height, width = image.shape
        watermark = self.generate_watermark(len(watermark_bits))
        watermarked_image = image.clone()
        positions = []
        
        # 1. DCT嵌入
        mid_band_pos = self.get_mid_band_positions(self.block_size)
        pos_idx = 0
        
        # 嵌入32位DCT水印
        for h in range(0, height, self.block_size):
            for w in range(0, width, self.block_size):
                h_end = min(h + self.block_size, height)
                w_end = min(w + self.block_size, width)
                
                if h_end - h < self.block_size or w_end - w < self.block_size:
                    continue
                    
                block = image[h:h_end, w:w_end]
                
                # 先将张量转换为 NumPy 数组，进行 DCT 变换
                block_np = block.numpy()
                dct_block = dct(dct(block_np.T, norm='ortho').T, norm='ortho')
                
                for idx, (bit, spread) in enumerate(zip(watermark_bits[pos_idx:], watermark[pos_idx:])):  
                    if idx >= len(mid_band_pos) or pos_idx >= 32:  # 只嵌入32位
                        break
                    
                    pos_h, pos_w = mid_band_pos[idx]
                    dct_block[pos_h, pos_w] += self.alpha * bit * spread
                    positions.append(('dct', h+pos_h, w+pos_w))
                    pos_idx += 1
                    
                    if pos_idx >= len(watermark_bits) or pos_idx >= 32:  # 只嵌入32位
                        break
                
                # 用逆 DCT 恢复水印嵌入的块
                watermarked_block = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
                
                # 重新将其转换为张量并保存到水印图像中
                watermarked_image[h:h_end, w:w_end] = torch.tensor(watermarked_block)
                
                if pos_idx >= 32:  # 只嵌入32位
                    break
            if pos_idx >= 32:  # 只嵌入32位
                break
        
        return watermarked_image, positions
    
    def extract(self, watermarked_image, positions, watermark_length):
        watermark = self.generate_watermark(watermark_length)
        extracted_values = []

        # 如果水印图像在GPU上，先转到CPU并转换为NumPy数组
        if watermarked_image.is_cuda:
            watermarked_image = watermarked_image.cpu()

        watermarked_image_np = watermarked_image.numpy()

        # 获取DWT系数
        cA, (cH, cV, cD) = self.dwt2(watermarked_image_np)

        # 分别提取DCT和DWT水印
        for i, pos in enumerate(positions):
            if pos[0] == 'dwt':
                _, band, i, j = pos
                if band == 'cH':
                    value = cH[i, j] * watermark[i]
            else:  # dct
                _, h, w = pos
                block_h = h // self.block_size * self.block_size
                block_w = w // self.block_size * self.block_size
                block = watermarked_image_np[block_h:block_h+self.block_size,
                                             block_w:block_w+self.block_size]

                # 计算DCT
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                value = dct_block[h-block_h, w-block_w] * watermark[i]

            extracted_values.append(value)

        # 使用单一阈值判决
        values_mean = np.mean(extracted_values)
        values_std = np.std(extracted_values)
        threshold = values_mean - 0.2 * values_std
        extracted_bits = [1 if v > threshold else 0 for v in extracted_values]

        return extracted_bits, extracted_values

def test_improved_watermark():
    # 水印信息
    watermark_bits = [1, 0, 1, 1, 0, 0, 1, 0] * 8  # 64位水印
    
    # 初始化水印器
    watermarker1 = HybridWatermark(block_size=16, alpha=0.5, beta=0.6)
    
    # 嵌入水印
    print(stacked_tensor)
    watermarked_tensor1, positions1 = watermarker1.embed(stacked_tensor, watermark_bits)
    print("获得嵌入水印之后的矩阵C")
    """
        ## 添加噪声
        后续可以把这里当作是对模型参数的微调，然后测试一下需要微调多久、时间、算力我们的水印才检测不出来了
        noisy_image = watermarked_image + np.random.normal(0, 0.005, watermarked_image.shape)
    """
    
    # 提取水印
    ## 输入： 待提取水印的张量阵（从端侧的O矩阵一步步得到的）
    ## 输入： 嵌入水印的位置（云侧预先存储好的）
    ## 输入： 水印比特序列长度（云侧预先存储好的）
    ## 返回值： 提取出的水印比特序列
    print("开始检测水印")
    extracted_bits1, values1 = watermarker1.extract(watermarked_tensor1, positions1, len(positions1))
    print("检测水印结束")
    mse1 = np.mean((watermarked_tensor1.numpy() - stacked_tensor.numpy()) ** 2)
    
    accuracy1 = sum(a == b for a, b in zip(watermark_bits[:len(positions1)], extracted_bits1)) / len(extracted_bits1)
    print("水印完整度:" + str(accuracy1)) 
restored_tensor1 = test_improved_watermark()


