# pip install connected-components-3d
import numpy as np
import SimpleITK as sitk
import os

join = os.path.join
from skimage import transform
from tqdm import tqdm
import cc3d

# convert nii image to npz files, including original image and corresponding masks
modality = "CT"
anatomy = "Abd"  # anatomy + dataset name
img_name_suffix = "_0000.nii.gz"
gt_name_suffix = ".nii.gz"
prefix = modality + "_" + anatomy + "_"

nii_path = "data/images"  # path to the nii images
gt_path = "data/labels"  # path to the ground truth
npy_path = "data/npy/" + prefix[:-1]
os.makedirs(join(npy_path, "gts"), exist_ok=True)
os.makedirs(join(npy_path, "imgs"), exist_ok=True)

image_size = 1024
voxel_num_thre2d = 50
voxel_num_thre3d = 500

names = sorted(os.listdir(gt_path))
print(f"ori # files {len(names)}")
names = [
    name
    for name in names
    if os.path.exists(join(nii_path, name.split(gt_name_suffix)[0] + img_name_suffix))
]
print(f"after sanity check # files {len(names)}")

# set label ids that are excluded
remove_label_ids = []

# set window level and width
# https://radiopaedia.org/articles/windowing-ct
WINDOW_LEVEL = 40  # only for CT images
WINDOW_WIDTH = 400  # only for CT images

# Define organ labels and names
organ_labels = [1, 2, 3, 4]  # 器官标签：1-肝脏，2-肾脏，3-脾脏，4-胰腺
organ_names = {1: 'liver', 2: 'kidney', 3: 'spleen', 4: 'pancreas'}

# %% save preprocessed images and masks as npz files
for organ_label in organ_labels:
    organ_name = organ_names[organ_label]
    print(f"Processing organ: {organ_name} (label={organ_label})")
    # 创建器官特定的文件夹
    organ_imgs_path = join(npy_path, "imgs", organ_name)
    organ_gts_path = join(npy_path, "gts", organ_name)
    os.makedirs(organ_imgs_path, exist_ok=True)
    os.makedirs(organ_gts_path, exist_ok=True)

    for name in tqdm(names):  # 遍历所有病例
        image_name = name.split(gt_name_suffix)[0] + img_name_suffix
        gt_name = name
        gt_sitk = sitk.ReadImage(join(gt_path, gt_name))
        gt_data_ori = np.uint8(sitk.GetArrayFromImage(gt_sitk))

        # remove label ids
        for remove_label_id in remove_label_ids:
            gt_data_ori[gt_data_ori == remove_label_id] = 0

        # Generate binary mask for the current organ
        gt_data_binary = np.uint8(gt_data_ori == organ_label)

        # Exclude objects with less than threshold voxels in 3D
        gt_data_binary = cc3d.dust(
            gt_data_binary, threshold=voxel_num_thre3d, connectivity=26, in_place=True
        )

        # Remove small objects in 2D slices
        for slice_i in range(gt_data_binary.shape[0]):
            gt_i = gt_data_binary[slice_i, :, :]
            gt_data_binary[slice_i, :, :] = cc3d.dust(
                gt_i, threshold=voxel_num_thre2d, connectivity=8, in_place=True
            )

        # Find non-zero slices
        z_index, _, _ = np.where(gt_data_binary > 0)
        z_index = np.unique(z_index)

        if len(z_index) > 0:
            # Crop the ground truth with non-zero slices
            gt_roi = gt_data_binary[z_index, :, :]
            # Load image and preprocess
            img_sitk = sitk.ReadImage(join(nii_path, image_name))
            image_data = sitk.GetArrayFromImage(img_sitk)
            # nii preprocess start
            if modality == "CT":
                lower_bound = WINDOW_LEVEL - WINDOW_WIDTH / 2
                upper_bound = WINDOW_LEVEL + WINDOW_WIDTH / 2
                image_data_pre = np.clip(image_data, lower_bound, upper_bound)
                image_data_pre = (
                    (image_data_pre - np.min(image_data_pre))
                    / (np.max(image_data_pre) - np.min(image_data_pre))
                    * 255.0
                )
            else:
                lower_bound, upper_bound = np.percentile(
                    image_data[image_data > 0], 0.5
                ), np.percentile(image_data[image_data > 0], 99.5)
                image_data_pre = np.clip(image_data, lower_bound, upper_bound)
                image_data_pre = (
                    (image_data_pre - np.min(image_data_pre))
                    / (np.max(image_data_pre) - np.min(image_data_pre))
                    * 255.0
                )
                image_data_pre[image_data == 0] = 0

            image_data_pre = np.uint8(image_data_pre)
            img_roi = image_data_pre[z_index, :, :]

            # Save preprocessed images and masks as npz files
            np.savez_compressed(
                join(npy_path, prefix + organ_name + '_' + gt_name.split(gt_name_suffix)[0]+'.npz'),
                imgs=img_roi, gts=gt_roi, spacing=img_sitk.GetSpacing()
            )

            # Save the image and ground truth as nii files for sanity check;
            img_roi_sitk = sitk.GetImageFromArray(img_roi)
            gt_roi_sitk = sitk.GetImageFromArray(gt_roi)
            sitk.WriteImage(
                img_roi_sitk,
                join(npy_path, prefix + organ_name + '_' + gt_name.split(gt_name_suffix)[0] + "_img.nii.gz"),
            )
            sitk.WriteImage(
                gt_roi_sitk,
                join(npy_path, prefix + organ_name + '_' + gt_name.split(gt_name_suffix)[0] + "_gt.nii.gz"),
            )

            # Save each CT image as npy file
            for i in range(img_roi.shape[0]):
                img_i = img_roi[i, :, :]
                img_3c = np.repeat(img_i[:, :, None], 3, axis=-1)
                resize_img_skimg = transform.resize(
                    img_3c,
                    (image_size, image_size),
                    order=3,
                    preserve_range=True,
                    mode="constant",
                    anti_aliasing=True,
                )
                resize_img_skimg_01 = (resize_img_skimg - resize_img_skimg.min()) / np.clip(
                    resize_img_skimg.max() - resize_img_skimg.min(), a_min=1e-8, a_max=None
                )  # normalize to [0, 1], (H, W, 3)

                gt_i = gt_roi[i, :, :]
                resize_gt_skimg = transform.resize(
                    gt_i,
                    (image_size, image_size),
                    order=0,
                    preserve_range=True,
                    mode="constant",
                    anti_aliasing=False,
                )
                resize_gt_skimg = np.uint8(resize_gt_skimg)
                assert resize_img_skimg_01.shape[:2] == resize_gt_skimg.shape

                # Save the image and ground truth slices with organ name in the path
                np.save(
                    join(
                        organ_imgs_path,
                        prefix
                        + organ_name + '_'
                        + gt_name.split(gt_name_suffix)[0]
                        + "-"
                        + str(i).zfill(3)
                        + ".npy",
                    ),
                    resize_img_skimg_01,
                )
                np.save(
                    join(
                        organ_gts_path,
                        prefix
                        + organ_name + '_'
                        + gt_name.split(gt_name_suffix)[0]
                        + "-"
                        + str(i).zfill(3)
                        + ".npy",
                    ),
                    resize_gt_skimg,
                )
