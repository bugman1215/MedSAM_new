import numpy as np
import os
from tqdm import tqdm
from PIL import Image

# 定义路径
base_path = "data/npy/flareCT_Abd"
imgs_path = os.path.join(base_path, "imgs")
gts_path = os.path.join(base_path, "gts")
output_imgs_path = os.path.join("data/npy/compress/imgs")
output_gts_path = os.path.join("data/npy/compress/gts")
os.makedirs(output_imgs_path, exist_ok=True)
os.makedirs(output_gts_path, exist_ok=True)

# 遍历每个器官文件夹
for organ_name in os.listdir(imgs_path):
    organ_imgs_path = os.path.join(imgs_path, organ_name)
    organ_gts_path = os.path.join(gts_path, organ_name)
    organ_output_imgs_path = os.path.join(output_imgs_path, organ_name)
    organ_output_gts_path = os.path.join(output_gts_path, organ_name)
    os.makedirs(organ_output_imgs_path, exist_ok=True)
    os.makedirs(organ_output_gts_path, exist_ok=True)

    print(f"Compressing data for {organ_name}...")

    # 获取每个病例的 case_id（去除切片编号和扩展名）
    case_ids = sorted(set(f.split("_")[-1].split("-")[0] for f in os.listdir(organ_imgs_path) if f.endswith(".jpg")))

    for case_id in tqdm(case_ids, desc=f"Processing {organ_name}"):
        img_slices = []
        gt_slices = []

        # 按切片索引加载该 case_id 的所有图像和标签
        slice_idx = 0
        while True:
            img_slice_path = os.path.join(organ_imgs_path, f"CT_Abd_{organ_name}_train_{case_id}-{str(slice_idx).zfill(3)}.jpg")
            gt_slice_path = os.path.join(organ_gts_path, f"CT_Abd_{organ_name}_train_{case_id}-{str(slice_idx).zfill(3)}.png")
            
            # 检查切片文件是否存在
            if not os.path.isfile(img_slice_path) or not os.path.isfile(gt_slice_path):
                break  # 没有更多切片，结束循环

            # 加载图像和标签切片
            img_slice = np.array(Image.open(img_slice_path)) / 255.0  # 归一化
            gt_slice = np.array(Image.open(gt_slice_path))

            img_slices.append(img_slice)
            gt_slices.append(gt_slice)
            slice_idx += 1

        # 将当前 case_id 的图像和标签切片分别保存为单个 npz 文件
        img_slices = np.array(img_slices, dtype=np.float32)
        gt_slices = np.array(gt_slices, dtype=np.uint8)

        img_output_file = os.path.join(organ_output_imgs_path, f"CT_Abd_{organ_name}_{case_id}.npz")
        gt_output_file = os.path.join(organ_output_gts_path, f"CT_Abd_{organ_name}_{case_id}.npz")
        np.savez_compressed(img_output_file, imgs=img_slices)
        np.savez_compressed(gt_output_file, gts=gt_slices)

        # 打印保存信息
        print(f"imgs {organ_name} {case_id} saved")
        print(f"gts {organ_name} {case_id} saved")

print("All files have been successfully compressed.")
