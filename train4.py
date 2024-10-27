# -*- coding: utf-8 -*-
"""
Train the image encoder and mask decoder with real-time loss and accuracy logging.
"""

# %% setup environment
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
import glob
from utils.SurfaceDice import compute_dice_coefficient  # Ensure utils.SurfaceDice is available

# Set seeds and cache
torch.manual_seed(2023)
torch.cuda.empty_cache()
join = os.path.join

# Set environment variables
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "6"


class NpyDataset(Dataset):
    def __init__(self, data_root, organ, bbox_shift=20, split="train"):
        self.data_root = data_root
        self.bbox_shift = bbox_shift
        self.organ = organ

        img_files = sorted(glob.glob(join(data_root, "imgs", organ, "*.npz")))
        gt_files = sorted(glob.glob(join(data_root, "gts", organ, "*.npz")))
        
        assert len(img_files) == len(gt_files), "Mismatch between img and gt files."
        
        total_files = len(img_files)
        split_idx = int(0.8 * total_files)
        if split == "train":
            self.img_files, self.gt_files = img_files[:split_idx], gt_files[:split_idx]
        else:
            self.img_files, self.gt_files = img_files[split_idx:], gt_files[split_idx:]
            
        print(f"Loaded {len(self.img_files)} .npz files for organ '{organ}' - split '{split}'.")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_file = self.img_files[index]
        gt_file = self.gt_files[index]

        img_data = np.load(img_file)["imgs"]
        gt_data = np.load(gt_file)["gts"]

        slice_idx = random.randint(0, img_data.shape[0] - 1)
        img_slice = img_data[slice_idx]
        gt_slice = gt_data[slice_idx]

        img_slice = np.transpose(img_slice, (2, 0, 1))
        label_ids = np.unique(gt_slice)[1:]
        gt2D = np.uint8(gt_slice == random.choice(label_ids.tolist()))

        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])

        return (
            torch.tensor(img_slice).float(),
            torch.tensor(gt2D[None, :, :]).long(),
            torch.tensor(bboxes).float(),
            os.path.basename(img_file),
        )


# %% Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument("-task_name", type=str, default="MedSAM-ViT-B")
parser.add_argument("-model_type", type=str, default="vit_b")
parser.add_argument("-checkpoint", type=str, default="work_dir/SAM/sam_vit_b_01ec64.pth")
parser.add_argument("-work_dir", type=str, default="./work_dir")
parser.add_argument("-num_epochs", type=int, default=1000)
parser.add_argument("-batch_size", type=int, default=2)
parser.add_argument("-num_workers", type=int, default=0)
parser.add_argument("-weight_decay", type=float, default=0.01)
parser.add_argument("-lr", type=float, default=0.0001)
parser.add_argument("-use_amp", action="store_true", default=False)
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()


# %% Model setup
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
                points=None, boxes=box_torch, masks=None)
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding, image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings,
            multimask_output=False)
        ori_res_masks = F.interpolate(
            low_res_masks, size=(image.shape[2], image.shape[3]), mode="bilinear", align_corners=False)
        return ori_res_masks


def main():
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    for organ in ["liver", "kidney", "spleen", "pancreas"]:
        print(f"Starting training for {organ}")
        
        # Set up data paths and DataLoader
        tr_npy_path = "data/npy/compress"
        
        train_dataset = NpyDataset(tr_npy_path, organ=organ, split="train")
        val_dataset = NpyDataset(tr_npy_path, organ=organ, split="val")

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        # Initialize model and set up paths
        model_save_path = join(args.work_dir, f"{args.task_name}_{organ}_{datetime.now().strftime('%Y%m%d-%H%M')}")
        os.makedirs(model_save_path, exist_ok=True)
        
        sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
        medsam_model = MedSAM(
            image_encoder=sam_model.image_encoder,
            mask_decoder=sam_model.mask_decoder,
            prompt_encoder=sam_model.prompt_encoder,
        ).to(device)
        medsam_model.train()

        optimizer = torch.optim.AdamW(
            list(medsam_model.image_encoder.parameters()) + list(medsam_model.mask_decoder.parameters()),
            lr=args.lr, weight_decay=args.weight_decay)
        
        seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
        ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
        
        best_loss = 1e10
        train_losses, val_losses, train_dices, val_dices = [], [], [], []
        scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
        
        for epoch in range(args.num_epochs):
            epoch_train_loss, epoch_train_dice = 0, 0
            medsam_model.train()
            for step, (image, gt2D, boxes, _) in enumerate(tqdm(train_dataloader)):
                optimizer.zero_grad()
                boxes_np = boxes.detach().cpu().numpy()
                image, gt2D = image.to(device), gt2D.to(device)

                if scaler:
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

                epoch_train_loss += loss.item()
                epoch_train_dice += compute_dice_coefficient(gt2D.cpu().numpy(), medsam_pred.detach().cpu().numpy() > 0.5)
            
            epoch_train_loss /= len(train_dataloader)
            epoch_train_dice /= len(train_dataloader)
            train_losses.append(epoch_train_loss)
            train_dices.append(epoch_train_dice)

            # Validation phase
            epoch_val_loss, epoch_val_dice = 0, 0
            medsam_model.eval()
            with torch.no_grad():
                for image, gt2D, boxes, _ in val_dataloader:
                    boxes_np = boxes.detach().cpu().numpy()
                    image, gt2D = image.to(device), gt2D.to(device)
                    medsam_pred = medsam_model(image, boxes_np)
                    val_loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
                    epoch_val_loss += val_loss.item()
                    epoch_val_dice += compute_dice_coefficient(gt2D.cpu().numpy(), medsam_pred.detach().cpu().numpy() > 0.5)
                
            epoch_val_loss /= len(val_dataloader)
            epoch_val_dice /= len(val_dataloader)
            val_losses.append(epoch_val_loss)
            val_dices.append(epoch_val_dice)

            # Print train and val metrics
            print(f"Epoch {epoch}, Train Loss: {epoch_train_loss}, Val Loss: {epoch_val_loss}, Train Dice: {epoch_train_dice}, Val Dice: {epoch_val_dice}")

            # Save latest and best models
            checkpoint = {
                "model": medsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            latest_path = join(model_save_path, "medsam_model_latest.pth")
            torch.save(checkpoint, latest_path)

            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                best_path = join(model_save_path, "medsam_model_best.pth")
                torch.save(checkpoint, best_path)
                print(f"Saved best model with val loss {epoch_val_loss} at {best_path}")

            # Real-time update of the metrics plot
            plt.plot(train_losses, label="Train Loss")
            plt.plot(val_losses, label="Validation Loss")
            plt.plot(train_dices, label="Train Dice")
            plt.plot(val_dices, label="Validation Dice")
            plt.title(f"Training and Validation Metrics for {organ}")
            plt.xlabel("Epoch")
            plt.ylabel("Metrics")
            plt.legend()
            loss_plot_path = join(model_save_path, f"{organ}_train_val_metrics.png")
            plt.savefig(loss_plot_path)
            plt.close()

if __name__ == "__main__":
    main()
