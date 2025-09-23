import random
from pathlib import Path
import torch
from tqdm import tqdm
import argparse
from algorithms.vae.common.distribution import DiagonalGaussianDistribution


def estimate_latent_stats(
    latent_dir: str,
    batch_size: int = 128,
    repeat: int = 5,
    channel_wise: bool = False,
    is_distribution: bool = False,
):
    data_dir = Path(latent_dir)
    latent_paths = [
        p for p in data_dir.glob("**/*.pt") if not p.name.endswith("_cond.pt")
    ]
    random.shuffle(latent_paths)
    means, stds = [], []

    for i in range(repeat):
        latents = None
        for idx in tqdm(
            range(batch_size * i, batch_size * (i + 1)), desc=f"Batch {i + 1}"
        ):
            latent = torch.load(latent_paths[idx])
            if is_distribution:
                latent = DiagonalGaussianDistribution(latent).sample()
            if latents is None:
                latents = latent
            else:
                latents = torch.cat([latents, latent], dim=0)

        if channel_wise:
            mean = latents.mean(dim=(0, 2, 3)).tolist()  # Compute mean for each channel
            std = latents.std(dim=(0, 2, 3)).tolist()  # Compute std for each channel
        else:
            mean = latents.mean().item()
            std = latents.std().item()

        means.append(mean)
        stds.append(std)
        print(f"Batch {i + 1}: mean: {mean}, std: {std}")

    if channel_wise:
        overall_mean = [
            sum(m[i] for m in means) / len(means) for i in range(len(means[0]))
        ]
        overall_std = [sum(s[i] for s in stds) / len(stds) for i in range(len(stds[0]))]
        overall_mean = [round(x, 3) for x in overall_mean]
        overall_std = [round(x, 3) for x in overall_std]
    else:
        overall_mean = sum(means) / len(means)
        overall_std = sum(stds) / len(stds)

    print(f"Overall mean: {overall_mean}, overall std: {overall_std}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dir", type=str, help="Path to the latent tensors")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--repeat", type=int, default=5, help="Number of repeats")
    parser.add_argument(
        "--channel_wise", action="store_true", help="Calculate statistics channel wise"
    )
    parser.add_argument(
        "--distribution", action="store_true", help="Latents are saved as distributions"
    )
    args = parser.parse_args()
    estimate_latent_stats(
        args.latent_dir,
        args.batch_size,
        args.repeat,
        args.channel_wise,
        args.distribution,
    )
