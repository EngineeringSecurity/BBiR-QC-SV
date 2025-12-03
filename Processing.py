import numpy as np
import os
from typing import Tuple
from PIL import Image
import math

class BinaryDataPreprocessorConfig:
    """预处理配置参数"""
    def __init__(self):
        # 采样参数
        self.sample_length = 4096  # 采样长度（数据块大小）

        # 图像参数
        self.image_size = (256, 256)  # 图像最终放缩的大小 (高度, 宽度)

        # 采样比例 (用于长文件)
        self.sampling_ratios = {
            'front': 0.2,    # 前部比例
            'middle': 0.6,   # 中部比例
            'back': 0.2      # 后部比例
        }

        # 文件扩展名
        self.input_extensions = ['.enc']  # 输入文件扩展名
        self.output_4kb_ext = '.bin'      # 4KB数据块输出扩展名
        self.output_image_ext = '.png'    # 图像输出扩展名


class BinaryDataPreprocessor:
    def __init__(self, config=None):
        """初始化预处理器"""
        self.config = config if config else BinaryDataPreprocessorConfig()

    def read_binary_file(self, file_path: str) -> bytes:
        """读取二进制文件"""
        with open(file_path, 'rb') as f:
            binary_data = f.read()
        return binary_data

    def smart_sampling(self, binary_data: bytes) -> bytes:
        """
        智能采样：将二进制数据转换为固定长度的数据块
        - 如果文件大小不足目标长度，则在中间插入0补足
        - 如果超过目标长度，则按照配置的比例采样
        """
        data_length = len(binary_data)
        target_length = self.config.sample_length

        if data_length == target_length:
            return binary_data

        if data_length < target_length:
            padding_size = target_length - data_length
            mid_point = data_length // 2
            padded_data = (binary_data[:mid_point] +
                         b'\x00' * padding_size +
                         binary_data[mid_point:])
            return padded_data

        # 长文件采样
        front_size = int(target_length * self.config.sampling_ratios['front'])
        back_size = int(target_length * self.config.sampling_ratios['back'])
        middle_size = target_length - front_size - back_size

        front_part = binary_data[:front_size]
        back_part = binary_data[-back_size:]

        middle_start = front_size
        middle_end = data_length - back_size
        middle_indices = np.linspace(middle_start, middle_end - 1, middle_size, dtype=int)
        middle_part = bytes(binary_data[idx] for idx in middle_indices)

        final_block = front_part + middle_part + back_part
        return final_block

    # ==========  以下为重写的 bit 级尾部采样版本  ==========
    def process_to_grayscale_image(self, binary_data: bytes) -> np.ndarray:
        """将二进制数据转换为灰度图并放缩到指定大小（bit级尾部等距采样填充）"""
        gray_values = np.frombuffer(binary_data, dtype=np.uint8)
        if len(gray_values) == 0:
            return np.zeros(self.config.image_size, dtype=np.uint8)

        total_bytes = len(gray_values)
        max_square_side = int(math.ceil(math.sqrt(total_bytes)))
        square_pixels_needed = max_square_side * max_square_side

        if total_bytes >= square_pixels_needed:
            sampled_values = gray_values[:square_pixels_needed]
        else:
            padding_pixels = square_pixels_needed - total_bytes
            padding_bits   = padding_pixels * 8

            # 原始数据 → bit 流
            bit_array = np.unpackbits(gray_values.view(np.uint8))

            # 在 bit 流里等距采样
            if len(bit_array) == 0:
                sampled_bits = np.zeros(padding_bits, dtype=np.uint8)
            else:
                step = len(bit_array) / padding_bits
                indices = np.floor(np.arange(padding_bits) * step).astype(int)
                indices = np.clip(indices, 0, len(bit_array) - 1)
                sampled_bits = bit_array[indices]

            # bit 流重新打包成字节
            pad_len = (len(sampled_bits) + 7) // 8 * 8
            sampled_bits = np.pad(sampled_bits, (0, pad_len - len(sampled_bits)), 'constant')
            padding_bytes = np.packbits(sampled_bits.view(np.uint8))

            sampled_values = np.concatenate([gray_values, padding_bytes])

        square_image = sampled_values.reshape((max_square_side, max_square_side))
        pil_image = Image.fromarray(square_image, mode='L')
        resized_image = pil_image.resize(
            (self.config.image_size[1], self.config.image_size[0]),
            Image.BILINEAR
        )
        return np.array(resized_image)
    # ==========  bit 级尾部采样结束  ==========

    def save_4kb_block(self, block_data: bytes, output_path: str):
        """保存数据块"""
        with open(output_path, 'wb') as f:
            f.write(block_data)

    def save_image(self, image: np.ndarray, output_path: str):
        """保存灰度图"""
        pil_image = Image.fromarray(image, mode='L')
        pil_image.save(output_path)

    def process_file(self, input_file: str, output_4kb_dir: str, output_image_dir: str) -> Tuple[bytes, np.ndarray]:
        """处理单个文件"""
        binary_data = self.read_binary_file(input_file)
        sampled_data = self.smart_sampling(binary_data)
        grayscale_image = self.process_to_grayscale_image(binary_data)

        base_name = os.path.splitext(os.path.basename(input_file))[0]

        #不再保存4KB采样数据
	#block_output_path = os.path.join(output_4kb_dir, f"{base_name}{self.config.output_4kb_ext}")
        #self.save_4kb_block(sampled_data, block_output_path)

        image_output_path = os.path.join(output_image_dir, f"{base_name}{self.config.output_image_ext}")
        self.save_image(grayscale_image, image_output_path)

        return sampled_data, grayscale_image

    def process_dataset(self, input_dir: str, output_4kb_dir: str, output_image_dir: str):
        """处理目录中的所有二进制文件，保持类别结构"""
        if not os.path.exists(input_dir):
            print(f"输入目录不存在: {input_dir}")
            return

        self._print_config_info()

        os.makedirs(output_4kb_dir, exist_ok=True)
        os.makedirs(output_image_dir, exist_ok=True)

        for class_name in os.listdir(input_dir):
            class_input_dir = os.path.join(input_dir, class_name)
            if not os.path.isdir(class_input_dir):
                continue

            class_4kb_dir = os.path.join(output_4kb_dir, class_name)
            class_image_dir = os.path.join(output_image_dir, class_name)
            os.makedirs(class_4kb_dir, exist_ok=True)
            os.makedirs(class_image_dir, exist_ok=True)

            binary_files = []
            for ext in self.config.input_extensions:
                ext_files = [f for f in os.listdir(class_input_dir)
                           if f.endswith(ext) and os.path.isfile(os.path.join(class_input_dir, f))]
                binary_files.extend(ext_files)

            print(f"处理类别 {class_name}: {len(binary_files)} 个文件")

            processed_count = 0
            for filename in binary_files:
                input_path = os.path.join(class_input_dir, filename)
                try:
                    self.process_file(input_path, class_4kb_dir, class_image_dir)
                    processed_count += 1
                except Exception as e:
                    print(f"处理文件 {input_path} 时出错: {str(e)}")
                    continue

            print(f"成功处理类别 {class_name}: {processed_count}/{len(binary_files)} 个文件")

    def _print_config_info(self):
        """打印配置信息"""
        print("=" * 50)
        print("预处理配置信息")
        print("=" * 50)
        print(f"采样长度: {self.config.sample_length} 字节")
        print(f"图像尺寸: {self.config.image_size[0]}x{self.config.image_size[1]}")
        print(f"采样比例: 前{self.config.sampling_ratios['front']*100}% / "
              f"中{self.config.sampling_ratios['middle']*100}% / "
              f"后{self.config.sampling_ratios['back']*100}%")
        print(f"输入文件扩展名: {', '.join(self.config.input_extensions)}")
        print(f"输出文件扩展名: 数据块={self.config.output_4kb_ext}, "
              f"图像={self.config.output_image_ext}")
        print("=" * 50)
        print("图像填充策略: 尾部等距采样填充（bit级）")
        print("=" * 50)


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    config = BinaryDataPreprocessorConfig()
    # 按需修改配置
    config.sample_length = 4096
    config.image_size = (256, 256)
    config.input_extensions = ['.txt']

    input_directory      = r"C:\Users\蒲\Desktop\谣言检测\实验数据\Twitter\twitter16txt"  # 输出目录
    output_4kb_directory = r"Y:\yaoyanjianceshujuji\DATASET\phemetxt-ban-4K"
    output_image_directory = r"C:\Users\蒲\Desktop\谣言检测\实验数据\Twitter\twitter16txt-image"  # 输出目录

    preprocessor = BinaryDataPreprocessor(config)
    preprocessor.process_dataset(input_directory,
                                 output_4kb_directory,
                                 output_image_directory)