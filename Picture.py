import numpy as np
from PIL import Image
import os

# 假设你的 .npz 文件路径为 'path_to_file.npz'
npz_file_path = "/Users/crisp/Downloads/wenjiecode/data/npy/CT_Abd/CT_Abd_FLARE22_Tr_0001.npz"

# 加载 .npz 文件
data = np.load(npz_file_path)

# 假设图像数据存储在键 'imgs' 中
images = data['imgs']  # 这可能是一个 3D 数组 (num_slices, height, width)

# 创建一个输出文件夹，用于保存png图片
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

# 遍历所有切片，将它们保存为png图片
for i, img_slice in enumerate(images):
    # 将切片数据归一化到0-255范围内 (如果是浮点数)
    if img_slice.max() > 1:  # 如果数值范围较大，则不做归一化
        img_slice = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice)) * 255.0

    # 转换为uint8格式
    img_slice_uint8 = img_slice.astype(np.uint8)

    # 使用 PIL 保存图片
    img = Image.fromarray(img_slice_uint8)
    img.save(os.path.join(output_dir, f"slice_{i}.png"))

    print(f"Saved slice_{i}.png")
