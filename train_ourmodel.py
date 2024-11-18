# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""

# %% setup environment
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict
join = os.path.join
from tqdm import tqdm
from transformers import AutoTokenizer
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob
from utils.SurfaceDice import compute_dice_coefficient

from get_clip_embedding1 import get_clip_embeddings
from get_clip_embedding1 import create_modified_clip_model



# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()

# torch.distributed.init_process_group(backend="gloo")

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

# -*- coding: utf-8 -*-
"""
Train the image encoder and mask decoder.
Freeze prompt image encoder.
"""


# Helper functions
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )

# Dataset class for loading .npz files
class NpyDataset(Dataset):
    def __init__(self, data_roots, bbox_shift=20, tokenizer_name="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"):
        """
        Dataset for loading .npz files of medical images and labels.

        Args:
            data_roots (str): Root directory containing 'imgs' and 'gts' subdirectories.
            bbox_shift (int): Random shift range for bounding box calculation.
            tokenizer_name (str): Huggingface tokenizer name for text descriptions.
        """
        self.data_roots = data_roots
        self.img_paths = []
        self.gt_paths = []
        self.text_descriptions = []

        self.bbox_shift = bbox_shift
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Iterate over organ directories
        for organ_name in ["liver", "kidney", "spleen", "pancreas"]:
            img_dir = os.path.join(data_roots, "imgs", organ_name)
            gt_dir = os.path.join(data_roots, "gts", organ_name)
            description_file = os.path.join(img_dir, "descriptions.txt")

            # Load organ description
            if os.path.isfile(description_file):
                with open(description_file, 'r') as f:
                    description = f.read().strip()
            else:
                description = f"{organ_name} medical image segmentation"

            # Collect all image and ground truth file paths
            organ_img_paths = sorted(glob.glob(os.path.join(img_dir, "*.npz")))
            organ_gt_paths = sorted(glob.glob(os.path.join(gt_dir, "*.npz")))

            assert len(organ_img_paths) == len(organ_gt_paths), f"Mismatch between images and labels for {organ_name}"

            self.img_paths.extend(organ_img_paths)
            self.gt_paths.extend(organ_gt_paths)
            self.text_descriptions.extend([description] * len(organ_img_paths))

        print(f"Number of images: {len(self.img_paths)}")

    def __len__(self):
        return len(self.gt_paths)

    def __getitem__(self, index):
        """
        Load a single image-label pair, compute bounding box, and tokenize description.

        Args:
            index (int): Index of the data sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, torch.Tensor]:
                - Image tensor of shape [C, H, W].
                - Ground truth tensor of shape [1, H, W].
                - Bounding box tensor of shape [4].
                - Image file name.
                - Tokenized text description tensor of shape [seq_len].
        """
        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]

        img_npz = np.load(img_path)
        gt_npz = np.load(gt_path)

        # Extract image and ground truth arrays
        img = img_npz["imgs"]  # Shape: [Slices, H, W, (C)] or [Slices, H, W]
        gt = gt_npz["gts"]     # Shape: [Slices, H, W]

        # Use the first slice for 2D training
        img_2D = img[0]  # Shape: [H, W, (C)] or [H, W]
        gt2D = np.uint8(gt[0] > 0)  # Binary mask

        # Ensure image has [C, H, W] format
        if len(img_2D.shape) == 3:  # If RGB [H, W, C]
            img_2D = np.transpose(img_2D, (2, 0, 1))  # Convert to [C, H, W]
        else:  # If grayscale [H, W]
            img_2D = img_2D[np.newaxis, :, :]  # Add channel dimension: [1, H, W]

        # Compute bounding box
        y_indices, x_indices = np.where(gt2D > 0)
        if len(x_indices) > 0:  # Ensure there are labeled regions
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            H, W = gt2D.shape
            x_min = max(0, x_min - random.randint(0, self.bbox_shift))
            x_max = min(W, x_max + random.randint(0, self.bbox_shift))
            y_min = max(0, y_min - random.randint(0, self.bbox_shift))
            y_max = min(H, y_max + random.randint(0, self.bbox_shift))
            bboxes = np.array([x_min, y_min, x_max, y_max])
        else:
            bboxes = np.array([0, 0, 0, 0])  # Default bounding box for empty regions

        # Tokenize the description
        description = self.text_descriptions[index]
        text_tokens = self.tokenizer(
            description,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=77
        )

        return (
            torch.tensor(img_2D).float(),
            torch.tensor(gt2D).unsqueeze(0).long(),  # Add channel dimension
            torch.tensor(bboxes).float(),
            os.path.basename(img_path),
            text_tokens['input_ids'].squeeze(0),
        )

# MedSAM model
class MedSAM(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

        # Freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box, text_input):
        image_embedding = self.image_encoder(image)
        device = image_embedding[0].device if isinstance(image_embedding, list) else image_embedding.device

        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None, boxes=box_torch, masks=None
            )
            clip_embeddings = get_clip_embeddings(image, text_input).to(device)

        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            clip_prompt_embeddings=clip_embeddings,
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks


def train_val_split(dataset, val_organs=["liver", "kidney", "spleen", "pancreas"]):
    """
    Split the dataset into training and validation subsets by organ.
    """
    val_indices = {organ: [] for organ in val_organs}
    train_indices = []

    for idx, path in enumerate(dataset.img_paths):
        organ_found = False
        for organ in val_organs:
            if f"/{organ}/" in path:  # 检查路径中是否包含器官文件夹
                # 将一部分器官数据分配到验证集
                if len(val_indices[organ]) < len(dataset.img_paths) // (5 * len(val_organs)):
                    val_indices[organ].append(idx)
                else:
                    train_indices.append(idx)
                organ_found = True
                break
        if not organ_found:
            train_indices.append(idx)

    # Debug: 打印数据集分配信息
    print(f"Training set size: {len(train_indices)}")
    for organ, indices in val_indices.items():
        print(f"Validation set for {organ}: {len(indices)}")

    if not train_indices:
        raise ValueError("Training set is empty! Check your dataset split logic.")

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_datasets = {
        organ: torch.utils.data.Subset(dataset, val_indices[organ]) for organ in val_organs if val_indices[organ]
    }

    return train_dataset, val_datasets

# Main training script
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--tr_npy_path", type=str, default="data/flare2021npy")
    parser = argparse.ArgumentParser()
    parser.add_argument("-task_name", type=str, default="MedSAM-ViT-B")
    parser.add_argument("-model_type", type=str, default="vit_b")
    parser.add_argument(
        "-checkpoint", type=str, default="work_dir/MedSAM-ViT-B-20241116-1651/medsam_model_latest.pth"
    )
    parser.add_argument("--load_pretrain", type=bool, default=True, help="Load pretrain model")
    parser.add_argument("-pretrain_model_path", type=str, default="")
    parser.add_argument("-work_dir", type=str, default="./work_dir")
    parser.add_argument("-num_epochs", type=int, default=100)
    parser.add_argument("-batch_size", type=int, default=4)
    parser.add_argument("-num_workers", type=int, default=2)
    parser.add_argument("-weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("-lr", type=float, default=0.0001, metavar="LR", help="Learning rate")
    parser.add_argument("-use_wandb", type=bool, default=False, help="Use wandb for training log")
    parser.add_argument("-use_amp", action="store_true", default=False, help="Use AMP")
    parser.add_argument("--resume", type=str, default="", help="Resume training from checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    join = os.path.join
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
    #args = parser.parse_args()
    device = torch.device(args.device)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
        print(f"Created directory: {model_save_path}")
    

    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(args.device)
    medsam_model.train()

    optimizer = torch.optim.AdamW(
        medsam_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True)
    ce_loss = nn.BCEWithLogitsLoss()
    full_dataset = NpyDataset("data/flare2021npy")
    train_dataset, val_datasets = train_val_split(full_dataset)

    print(f"Number of training samples: {len(train_dataset)}")
    for organ, val_dataset in val_datasets.items():
        print(f"Validation set for {organ}: {len(val_dataset)}")
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    checkpoint_path = args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# Ensure 'state_dict' is extracted if the checkpoint contains additional metadata
    if 'model' in checkpoint:
        state_dict = checkpoint['model']  # Adjust this key based on the actual checkpoint structure
    else:
        state_dict = checkpoint

# Load the extracted state_dict into the model
    sam_model.load_state_dict(state_dict, strict=False)

# Optionally, extract the starting epoch if available
    start_epoch = checkpoint.get('epoch', 0)
    start_epoch+=1

    print(f"Checkpoint loaded successfully. Starting from epoch {start_epoch}.")
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
        #Store training and validation metrics
    train_losses = []
    train_accuracy = []
    val_losses = {organ: [] for organ in val_datasets.keys()}
    val_accuracies = {organ: [] for organ in val_datasets.keys()}
    start_epoch =0
    num_epochs=100
    best_loss = 1e10
    iter_num=0

        


    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        epoch_dice = 0
        for step, (image, gt2D, boxes, img_name, text_input) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            boxes_np = boxes.detach().cpu().numpy()
            image, gt2D = image.to(device), gt2D.to(device)
            if args.use_amp:
                ## AMP
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    medsam_pred = medsam_model(image, boxes_np)
                    loss = seg_loss(medsam_pred, gt2D) + ce_loss(
                        medsam_pred, gt2D.float()
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                medsam_pred = medsam_model(image, boxes_np, text_input)
                loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            dice_coefficient = compute_dice_coefficient(gt2D.detach().cpu().numpy(), medsam_pred.detach().cpu().numpy() > 0.5 )
            epoch_dice += dice_coefficient

            epoch_loss += loss.item()
            print(epoch_loss)
            iter_num += 1

        epoch_loss /= step
        epoch_dice /= step
        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_dice)

        if args.use_wandb:
            wandb.log({"epoch_loss": epoch_loss}, {"epoch_dice": epoch_dice})# Validation loop

        medsam_model.eval()
        val_results = defaultdict(dict)
        with torch.no_grad():
            for organ, val_dataset in val_datasets.items():
                val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
                total_loss, total_dice, count = 0, 0, 0

        # 使用 tqdm 显示进度条
                for image, gt, boxes, _, text_input in tqdm(val_loader, desc=f"Validating {organ}", leave=False):
                    bboxes = boxes.numpy()
                    image, gt = image.to(device), gt.to(device)

            # 模型预测
                    medsam_pred = medsam_model(image, bboxes, text_input)
                    loss = seg_loss(medsam_pred, gt) + ce_loss(medsam_pred, gt.float())
                    dice = compute_dice_coefficient(
                        gt.cpu().numpy(), (medsam_pred > 0.5).cpu().numpy()
                    )

            # 累加损失和指标
                    total_loss += loss.item()
                    total_dice += dice
                    count += 1

        # 计算平均损失和指标
                val_results[organ]["loss"] = total_loss / count
                val_results[organ]["dice"] = total_dice / count
                val_losses[organ].append(val_results[organ]["loss"])
                val_accuracies[organ].append(val_results[organ]["dice"])
                print(f"{organ.capitalize()} Validation Complete: Loss: {val_results[organ]['loss']:.4f}, Dice: {val_results[organ]['dice']:.4f}")

                
           
        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, train_Loss: {epoch_loss}, train_accuracy : {epoch_dice}'
        )
        # Logging metrics
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        # print(f"Train Loss: {avg_train_loss:.4f}")
        for organ, metrics in val_results.items():
            print(f"  {organ.capitalize()} - Val Loss: {metrics['loss']:.4f}, Dice: {metrics['dice']:.4f}")

        ## save the latest model
        checkpoint = {
            "model": medsam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, join(model_save_path, "medsam_model_latest.pth"))
        ## save the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint = {
                "model": medsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(model_save_path, "medsam_model_best.pth"))

        # %% plot loss
    
        # Update loss plots
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label="Train Loss")
        for organ, losses in val_losses.items():
            plt.plot(losses, label=f"Val Loss ({organ})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_accuracy, label="Train accuracy")
        for organ, accuracies in val_accuracies.items():
            plt.plot(accuracies, label=f"Val Dice ({organ})")
        plt.xlabel("Epoch")
        plt.ylabel("Dice Score")
        plt.title("Validation Dice Score")
        plt.legend()

        plt.tight_layout()
        plt.savefig(join(model_save_path, "training_curves.png"))
        plt.close()


        


if __name__ == "__main__":
    main()
