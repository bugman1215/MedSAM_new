# -*- coding: utf-8 -*-
"""
Train the image encoder and mask decoder
Freeze prompt image encoder
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
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
from collections import defaultdict
from utils.SurfaceDice import compute_dice_coefficient

# Set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "6"

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--tr_npy_path",
    type=str,
    default="/projects/p32500/wenjie3/data/flare2021npy",
    help="Path to training npy files; two subfolders: gts and imgs",
)
parser.add_argument("-task_name", type=str, default="MedSAM-ViT-B")
parser.add_argument("-model_type", type=str, default="vit_b")
parser.add_argument(
    "-checkpoint", type=str, default="work_dir/MedSAM-ViT-B-20241115-0247/medsam_model_latest.pth"
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

# Dataset class
class NpyDataset(Dataset):
    def __init__(self, data_roots, bbox_shift=20):
        self.data_roots = data_roots
        self.img_paths = []
        self.gt_paths = []
        self.bbox_shift = bbox_shift

        for organ_name in ["liver", "kidney", "spleen", "pancreas"]:
            img_dir = os.path.join(data_roots, "imgs", organ_name)
            gt_dir = os.path.join(data_roots, "gts", organ_name)
            organ_img_paths = sorted(glob.glob(os.path.join(img_dir, "*.npz")))
            organ_gt_paths = sorted(glob.glob(os.path.join(gt_dir, "*.npz")))

            assert len(organ_img_paths) == len(organ_gt_paths), f"Mismatch between images and labels for {organ_name}"

            self.img_paths.extend(organ_img_paths)
            self.gt_paths.extend(organ_gt_paths)

        print(f"Number of images: {len(self.img_paths)}")

    def __len__(self):
        return len(self.gt_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]

        img_npz = np.load(img_path)
        gt_npz = np.load(gt_path)

        img = img_npz["imgs"]
        gt = gt_npz["gts"]

        img_2D = img[0]
        gt2D = np.uint8(gt[0] > 0)

        if len(img_2D.shape) == 3:
            img_2D = np.transpose(img_2D, (2, 0, 1))
        else:
            img_2D = img_2D[np.newaxis, :, :]

        #img_2D = (img_2D - np.min(img_2D)) / (np.max(img_2D) - np.min(img_2D))

        y_indices, x_indices = np.where(gt2D > 0)
        if len(x_indices) > 0:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            H, W = gt2D.shape
            x_min = max(0, x_min - random.randint(0, self.bbox_shift))
            x_max = min(W, x_max + random.randint(0, self.bbox_shift))
            y_min = max(0, y_min - random.randint(0, self.bbox_shift))
            y_max = min(H, y_max + random.randint(0, self.bbox_shift))
            bboxes = np.array([x_min, y_min, x_max, y_max])
        else:
            bboxes = np.array([0, 0, 0, 0])

        return (
            torch.tensor(img_2D).float(),
            torch.tensor(gt2D).unsqueeze(0).long(),
            torch.tensor(bboxes).float(),
            os.path.basename(img_path),
        )
join = os.path.join
run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
device = torch.device(args.device)
# Model definition
class MedSAM(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding = self.image_encoder(image)
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None, boxes=box_torch, masks=None
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks

# Training and validation split
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



# Validation
def validate_model(model, val_datasets, seg_loss, ce_loss, device):
    model.eval()
    val_results = defaultdict(dict)
    with torch.no_grad():
        for organ, dataset in val_datasets.items():
            val_loader = DataLoader(dataset, batch_size=1, shuffle=False)
            total_loss, total_dice, count = 0, 0, 0

            for image, gt, boxes, _ in val_loader:
                image, gt = image.to(device), gt.to(device)
                boxes_np = boxes.cpu().numpy()
                pred = model(image, boxes_np)
                loss = seg_loss(pred, gt) + ce_loss(pred, gt.float())
                dice = compute_dice_coefficient(gt.cpu().numpy(), (pred > 0.5).cpu().numpy())

                total_loss += loss.item()
                total_dice += dice
                count += 1

            val_results[organ]["loss"] = total_loss / count
            val_results[organ]["dice"] = total_dice / count

    return val_results

# Plot training and validation results
def plot_training_curves(train_losses, val_losses, val_accuracies, save_path):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    for organ, losses in val_losses.items():
        plt.plot(losses, label=f"Val Loss ({organ})")
    plt.legend()
    plt.title("Loss")
    plt.subplot(1, 2, 2)
    for organ, acc in val_accuracies.items():
        plt.plot(acc, label=f"Dice ({organ})")
    plt.legend()
    plt.title("Accuracy")
    plt.savefig(save_path)
    plt.close()

def main():
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )

    # Load SAM model
    sam_model = sam_model_registry[args.model_type]("work_dir/SAM/sam_vit_b_01ec64.pth")
        # Load the checkpoint file
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

    print(f"Checkpoint loaded successfully. Starting from epoch {start_epoch}.")
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    medsam_model.train()

    print(
        "Number of total parameters: ",
        sum(p.numel() for p in medsam_model.parameters()),
    )  # 93735472
    print(
        "Number of trainable parameters: ",
        sum(p.numel() for p in medsam_model.parameters() if p.requires_grad),
    )  # 93729252

    img_mask_encdec_params = list(medsam_model.image_encoder.parameters()) + list(
        medsam_model.mask_decoder.parameters()
    )
    optimizer = torch.optim.AdamW(
        img_mask_encdec_params, lr=args.lr, weight_decay=args.weight_decay
    )
    print(
        "Number of image encoder and mask decoder parameters: ",
        sum(p.numel() for p in img_mask_encdec_params if p.requires_grad),
    )

    # Define loss functions
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    # Split dataset into train and val
    full_dataset = NpyDataset(args.tr_npy_path)
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

    
    if args.resume is not None and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        start_epoch = checkpoint["epoch"] + 1
        medsam_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    # Store training and validation metrics
    train_losses = []
    val_losses = {organ: [] for organ in val_datasets.keys()}
    val_accuracies = {organ: [] for organ in val_datasets.keys()}

    for epoch in range(start_epoch, args.num_epochs):
        # Training loop
        medsam_model.train()
        epoch_loss = 0
        for step, (image, gt2D, boxes, _) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            boxes_np = boxes.detach().cpu().numpy()
            image, gt2D = image.to(device), gt2D.to(device)

            if args.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    medsam_pred = medsam_model(image, boxes_np)
                    loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                medsam_pred = medsam_model(image, boxes_np)
                loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # Validation loop
        medsam_model.eval()
        val_results = defaultdict(dict)
        with torch.no_grad():
            for organ, val_dataset in val_datasets.items():
                val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
                total_loss, total_dice, count = 0, 0, 0

                for image, gt2D, boxes, _ in val_loader:
                    boxes_np = boxes.detach().cpu().numpy()
                    image, gt2D = image.to(device), gt2D.to(device)

                    medsam_pred = medsam_model(image, boxes_np)
                    loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
                    dice = compute_dice_coefficient(
                        gt2D.cpu().numpy(), (medsam_pred > 0.5).cpu().numpy()
                    )

                    total_loss += loss.item()
                    total_dice += dice
                    count += 1

                val_results[organ]["loss"] = total_loss / count
                val_results[organ]["dice"] = total_dice / count
                val_losses[organ].append(val_results[organ]["loss"])
                val_accuracies[organ].append(val_results[organ]["dice"])

        # Logging metrics
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        for organ, metrics in val_results.items():
            print(f"  {organ.capitalize()} - Val Loss: {metrics['loss']:.4f}, Dice: {metrics['dice']:.4f}")

        # Save model
        checkpoint = {
            "model": medsam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, join(model_save_path, "medsam_model_latest.pth"))
        if avg_train_loss < min(train_losses[:-1], default=avg_train_loss):
            torch.save(checkpoint, join(model_save_path, "medsam_model_best.pth"))

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
