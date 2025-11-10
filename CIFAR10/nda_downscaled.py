import os
import pickle
import random
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import math
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib import rcParams
import accelerate
import datasets
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from torchvision import transforms
from tqdm.auto import tqdm
from torch.utils.data import Dataset
import diffusers
from diffusers import DDPMScheduler

## set the seeds
def set_seeds(seed):
    set_seed(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that HF Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ddpm-model-64",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        default=False,
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main"
            " process."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
      "--gen_index_path",
      type=str,
      default=None,
      help="Path to a .pkl file containing selected indices for gen_dataset",
    )
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    
    parser.add_argument(
        "--index_path",
        type=str,
        default=None,
        help="TBD",
    )

    parser.add_argument(
        "--gen_path",
        type=str,
        default=None,
        help="TBD",
    )
    
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    
    parser.add_argument(
        "--t_strategy",
        type=str,
        default=None,
        help="TBD",
    )

    parser.add_argument(
       "--gen_source",
       type=str,
       default="gen",
       help="If set to 'idx-val', will load from dataset test split using val indices; otherwise, use image folder + gen_index_path.",
    )

    parser.add_argument("--e_seed", type=int, default=0, help="A seed for reproducible training.")

    parser.add_argument(
        "--save_vis_dir",
        type=str,
        default="topk_vis",
       help="Directory to save top-k visualization images"
    )

    parser.add_argument(
        "--K", 
        type=int, 
        default=10,
        help="Number of timesteps in uniform strategy"
    )

    parser.add_argument(
       "--t_fixed", 
       type=int, 
       default=None, 
       help="If set, only compute similarity at this fixed timestep"
    )

    parser.add_argument(
       "--patch_size", 
       type=int, 
       default=21, 
       help="Patch size (assumes square patch). Only used if --t_fixed is set"
    )

    parser.add_argument(
       "--weight_topk", 
       type=int, 
       default=None, 
       help="If set, use top-K weights instead of full sum during patch matching"
    )

    parser.add_argument(
       "--kernel_batch_size",
       type=int,
       default=16,
       help="Number of patch-kernels processed in one convolution call (higher is faster but uses more GPU memory)."
    )

    parser.add_argument(
       "--cache_dir",
       type=str,
       default=None,
       help="Where to cache HF datasets/models."
    )

    parser.add_argument("--gen_start", type=int, default=0, help="Start index for generated samples")
    parser.add_argument("--gen_end", type=int, default=None, help="End index for generated samples (exclusive)")

    parser.add_argument(
        "--proj_dim",
        type=int,
        default=None,
        help="Target downscaled patch side length s used in the projection. Must divide k (the patch_size side). "
         "If None, we will infer outside the function (odd k→(k-1)//2, even k→k//2)."
    )

    parser.add_argument(
       "--mask_value",
       type=float,
       default=1e3,
       help="Mask value to add on invalid patch locations for matching."
    )
   
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("You must specify either a dataset name from the hub or a train data directory.")

    return args

class IndexedDataset(Dataset):
    def __init__(self, dataset, global_indices):
        self.dataset = dataset
        self.global_indices = global_indices

    def __getitem__(self, idx):
        item = self.dataset[idx]
        item["global_idx"] = self.global_indices[idx]
        return item
    
    def __len__(self):
        return len(self.dataset)


def main():
    args = parse_args()
    print(args)

    ## initialize on the single gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    rank, world_size = 0, 1
    print(f"[Single GPU] Device: {device}")

    if args.seed is not None:
        set_seeds(args.seed + rank)

    train_dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir = args.cache_dir,
            split="train",
        )

    sub_idx = list(range(len(train_dataset)))
    print(sub_idx[0:5])
    train_dataset = train_dataset.select(sub_idx)

    if args.gen_source == "val":
        idx_val_path = args.index_path.replace("idx-train.pkl", "idx-val.pkl")
        gen_dataset = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        cache_dir=args.cache_dir,
        split="test",
    )
        with open(idx_val_path, 'rb') as f:
            val_idx = pickle.load(f)
        gen_dataset = gen_dataset.select(val_idx)
    
    else:
        import pandas as pd
        df = pd.DataFrame()
        df['path'] = ['{}/{}.png'.format(args.gen_path, i) for i in range(1000)]

        from datasets import DatasetDict, Dataset, Image
        gen_dataset = DatasetDict({
        "train": Dataset.from_dict({
            "img": df['path'].tolist(),
        }).cast_column("img", Image()),})
        gen_dataset = gen_dataset["train"]

        if args.gen_index_path is not None:
            with open(args.gen_index_path, 'rb') as f:
                test_index = pickle.load(f)
            print(f"Loaded {len(test_index)} test indices for gen_dataset.")
            gen_dataset = gen_dataset.select(test_index)
    

    augmentations = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform_images(examples):
        images = [augmentations(image.convert("RGB")) for image in examples["img"]]
        return {"input": images}
    
    train_dataset.set_transform(transform_images)
#    train_dataset = IndexedDataset(train_dataset, sub_idx)
    gen_dataset.set_transform(transform_images)

    if args.gen_end is not None:
       gen_dataset = gen_dataset.select(range(args.gen_start, args.gen_end))
       print(len(gen_dataset))
    

    train_dataloader = torch.utils.data.DataLoader(
       train_dataset,
       shuffle=False,
       batch_size=args.train_batch_size,
       num_workers=args.dataloader_num_workers,
    )

    all_images = []
    for batch in train_dataloader:
        all_images.append(batch["input"])
    y_full = torch.cat(all_images, 0).to(device, dtype)   # [N,C,H,W]  → FP16

    num_train = len(train_dataset)
    start_idx, end_idx = 0, num_train
    
    index_map = {orig_idx: i for i, orig_idx in enumerate(sub_idx)}   

    def extract_patches(img, patch_size=(3, 3), stride=(1, 1), dilation=(1, 1), mask_value=1e3):
        b, c, h, w = img.shape
        padding = (
          ((patch_size[0] - 1) * dilation[0]) // 2,
          ((patch_size[1] - 1) * dilation[1]) // 2,
        )
        img_padded = F.pad(img, (padding[1], padding[1], padding[0], padding[0]), value=0)
        unfold = F.unfold(img_padded, kernel_size=patch_size, stride=stride, dilation=dilation)
        h_out = (h + 2 * padding[0] - (patch_size[0] - 1) * dilation[0] - 1) // stride[0] + 1
        w_out = (w + 2 * padding[1] - (patch_size[1] - 1) * dilation[1] - 1) // stride[1] + 1

        # Shape: [B, H, W, C * K_h * K_w]
        patches = unfold.view(b, -1, h_out, w_out).permute(0, 2, 3, 1)
        # === Create binary mask patches in the same shape ===
        mask = torch.ones((b, c, h, w), device=img.device, dtype=dtype)
        mask = F.pad(mask, (padding[1], padding[1], padding[0], padding[0]), value=0)
        mask = F.unfold(mask, kernel_size=patch_size, stride=stride, dilation=dilation)
    
        # Also: [B, H, W, C * K_h * K_w]
        mask_patches = mask_value * (1 - (mask > 0).float())  # zeros where valid
        mask_patches = mask_patches.view(b, -1, h_out, w_out).permute(0, 2, 3, 1)
        
        return patches, mask_patches

    
    def get_projection_matrix_crop(k: int, s: int) -> torch.Tensor:
        assert k >= s and k % s == 0
        ratio = k // s
        P = torch.zeros((s * s, k * k), dtype=torch.float32)
        for i in range(s):
            for j in range(s):
                row_idx = i * s + j
                for di in range(ratio):
                    for dj in range(ratio):
                        src_i = i * ratio + di
                        src_j = j * ratio + dj
                        col_idx = src_i * k + src_j
                        P[row_idx, col_idx] = 1.0 / 2
        return P
    

    def chunked_y_norm_proj_N(y_eff: torch.Tensor,
                        P: torch.Tensor,
                        kH: int, kW: int,
                        stride=(1,1),
                        dilation=(1,1),
                        chunk_N: int = 8,
                        dtype=torch.float32):
        device = y_eff.device
        N, C, H_in, W_in = y_eff.shape
        H_out = (H_in - (kH - 1) * dilation[0] - 1) // stride[0] + 1
        W_out = (W_in - (kW - 1) * dilation[1] - 1) // stride[1] + 1

        out = torch.empty((N, H_out, W_out), device=device, dtype=dtype)

        for n0 in range(0, N, chunk_N):
            n1 = min(n0 + chunk_N, N)
            y_chunk = y_eff[n0:n1]                       # [Nc, C, H, W]
            y_unfold = F.unfold(y_chunk, kernel_size=(kH, kW), stride=stride, dilation=dilation)       # [Nc, C*k*k, H_out*W_out]
            Nc = y_unfold.size(0)
            k2 = kH * kW
            y_unfold = y_unfold.view(Nc, C, k2, -1).permute(0, 3, 1, 2)
            y_proj = torch.matmul(y_unfold, P.T)         # [Nc, L, C, s*s]
            y_norm_proj = (y_proj ** 2).sum(dim=(2, 3))  # [Nc, L]
            out[n0:n1] = y_norm_proj.view(Nc, H_out, W_out)
        return out
    

    def patch_score_model_helper_conv_softmax(
        x_patches, y_full, s, std,
        patch_size=(7, 7), stride=(1, 1), dilation=(1, 1),
        mask_value=1e3, kernel_batch_size=16, dtype=torch.float32, topk=None, proj_dim=None):

        y_full = y_full.to(dtype)
        x_patches = x_patches.to(dtype)
        B_x, H, W, patch_dim = x_patches.shape
        N, C, H_y, W_y = y_full.shape
        kH, kW = patch_size

        if proj_dim is None:
            assert kH == kW, "Only square patches supported for automatic proj_dim"
            proj_dim = (kH - 1) // 2 if kH % 2 == 1 else kH // 2

        pad_h = ((kH - 1) * dilation[0]) // 2
        pad_w = ((kW - 1) * dilation[1]) // 2
        y_pad = F.pad(y_full, (pad_w, pad_w, pad_h, pad_h), value=0)
        mask = torch.ones((N, C, H_y, W_y), device=y_full.device, dtype=dtype)
        mask = F.pad(mask, (pad_w, pad_w, pad_h, pad_h), value=0)
        mask_image = mask_value * (1 - mask)
        y_eff = s * y_pad + mask_image

        H_out = (H_y + 2 * pad_h - (kH - 1) * dilation[0]) // stride[0]
        W_out = (W_y + 2 * pad_w - (kW - 1) * dilation[1]) // stride[1]

        # === Projection Matrix ===
        P = get_projection_matrix_crop(kH, proj_dim).to(x_patches.device, dtype=dtype)  # [s*s, k*k]
        y_norm_proj = chunked_y_norm_proj_N(
            y_eff, P, kH, kW, stride, dilation,
            chunk_N=100,
            dtype=dtype)
        
        x_kernels = x_patches.reshape(B_x * H * W, C, kH, kW)
        weight_chunks = []
        for start in range(0, B_x * H * W, kernel_batch_size):
            end = min(start + kernel_batch_size, B_x * H * W)
            kernels_batch = x_kernels[start:end]
            B, C, kH, kW = kernels_batch.shape
            kernels_batch_flat = kernels_batch.reshape(B * C, kH * kW)        # [B*C, k*k]
            kernel_proj_flat = kernels_batch_flat @ P.T                    # [B*C, s*s]
            kernel_proj = kernel_proj_flat.reshape(B, C, -1)                  # [B, C, s*s]

            kernel_recon = (kernel_proj_flat @ P).reshape(B, C, kH, kW)
            # calculate |P^Tx|^2
            x_norm_batch = (kernel_proj ** 2).sum(dim=(1, 2))  #[Bp,]

            # convolve with recon kernel
            conv_out = F.conv2d(y_eff, kernel_recon, stride=stride, dilation=dilation)  #[]
            bp = kernels_batch.size(0)
            sim_map = conv_out

            sq_dist = (
               x_norm_batch.reshape(bp, 1, 1, 1) +         # broadcast
               y_norm_proj.reshape(1, N, H_out, W_out) -
               2 * sim_map.permute(1, 0, 2, 3)          # (bP,N,H_y,W_y)
            ).clamp(min=0)

            # --- softmax over (N·H_y·W_y) ---
            logits   = (-sq_dist / (2 * std**2)).flatten(1)      # (bP, N*H_y*W_y)
        #    print("logits.shape:", logits.shape)
            weight   = torch.softmax(logits, dim=-1).view(bp, N, H_out, W_out)
            if topk is not None:
               weight_flat = weight.view(bp, N, -1)                     # (bP, N, H_y*W_y)
               topk_val, _ = torch.topk(weight_flat, topk, dim=2)
               weight_sum_batch = topk_val.sum(dim=2)   
            else:
                weight_sum_batch = weight.sum(dim=(2, 3))  
            weight_chunks.append(weight_sum_batch)

        weight_all = torch.cat(weight_chunks, dim=0).view(B_x, H, W, N)
        weight = weight_all.sum(dim=(1, 2))
        return weight
        
    
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.ddpm_num_steps, beta_schedule=args.ddpm_beta_schedule)
    
    if args.t_fixed is not None:
        selected_timesteps = [args.t_fixed]
    else:
        if args.t_strategy=='uniform':
            selected_timesteps = range(0, 1000, 1000//args.K)
        elif args.t_strategy=='cumulative':
            selected_timesteps = range(0, args.K)       

    num_gen = len(gen_dataset)
    num_train = len(train_dataset)
    score_tensor = torch.zeros(len(selected_timesteps), num_gen, num_train, device=device, dtype=dtype)
    
    
    for gen_idx, gen_example in enumerate(tqdm(gen_dataset)):

        for t_idx, t in enumerate(selected_timesteps):
            set_seeds(args.e_seed * 1000 + t)
            x = gen_example["input"].unsqueeze(0).to(device, dtype)
            noise = torch.randn_like(x, dtype=dtype)
            bsz = x.shape[0]
            t = torch.tensor([int(t)] * bsz, device=x.device, dtype=torch.long)
    
            noisy_x = noise_scheduler.add_noise(x, noise, t)

            alpha_bar = noise_scheduler.alphas_cumprod[t.cpu()].item()
            s   = torch.sqrt(torch.tensor(alpha_bar, device=device, dtype=dtype))
            std = torch.sqrt(torch.tensor(1.0-alpha_bar, device=device, dtype=dtype))
            
            ## adaptive ps
            patch_size = (args.patch_size, args.patch_size)
                                
            x_patches, x_mask = extract_patches(
                noisy_x,
                patch_size=patch_size,
                mask_value=args.mask_value,
            )

            x_patches = x_patches + x_mask

            with torch.no_grad():
                weight_sum = patch_score_model_helper_conv_softmax(
                        x_patches, y_full, s, std,
                        patch_size=patch_size,
                        kernel_batch_size=args.kernel_batch_size,
                        dtype=dtype,
                        topk=args.weight_topk,
                        mask_value=args.mask_value,
                        proj_dim=args.proj_dim,
                    )               
            
            weight_full = weight_sum.squeeze(0)                # (num_train,)
            score_tensor[t_idx, gen_idx] = weight_full

    def visualize_topk_per_timestep(score_tensor, gen_dataset, train_dataset, save_dir, topk=10, timesteps=None):
        os.makedirs(save_dir, exist_ok=True)
    
        rcParams.update({
           'font.size': 10,
           'axes.titlesize': 10,
           'axes.labelsize': 10,
           'xtick.labelsize': 8,
           'ytick.labelsize': 8
        })

        num_timesteps = score_tensor.shape[0]
        num_gen = score_tensor.shape[1]

        for t_idx in range(num_timesteps):
            t = timesteps[t_idx] if timesteps else t_idx
            scores = score_tensor[t_idx]  # (num_gen, num_train)

            for i in range(num_gen):
                topk_idx = torch.argsort(scores[i], descending=True)[:topk]
                plot_images = [gen_dataset[i]['input']]
                plot_images.extend([train_dataset[int(idx)]['input'] for idx in topk_idx])
                original_indices = [sub_idx[int(idx)] for idx in topk_idx]
                topk_scores = [scores[i][idx].item() for idx in topk_idx]

                fig, axs = plt.subplots(1, topk + 1, figsize=(2 * (topk + 1), 2))
                for j, (ax, img) in enumerate(zip(axs, plot_images)):
                    ax.axis('off')
                    if j == 0:
                       ax.set_title(f'Gen {i}\nT={t}')
                    else:
                       ax.set_title(f'sub_idx={original_indices[j-1]}\n{topk_scores[j-1]:.2f}')
                    img = (img * 0.5 + 0.5).clamp(0, 1)
                    img = img.permute(1, 2, 0).cpu().numpy()  # <- Fix here
                    ax.imshow(img)
                plt.tight_layout()
                save_path = os.path.join(save_dir, f't{t}_gen{i}_topk.png')
                plt.savefig(save_path, dpi=150)
                plt.close()

    if rank == 0:
        filename = os.path.join(
            f"{args.output_dir}/scores-{args.e_seed}/score-{args.gen_source}{args.gen_start}_{args.gen_end}.npy")

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.save(filename, score_tensor.cpu().numpy())
        
        visualize_topk_per_timestep(
           score_tensor,
           gen_dataset,
           train_dataset,
           save_dir=args.save_vis_dir,
           topk=10,
           timesteps=selected_timesteps
           )


if __name__ == "__main__":
    main()
