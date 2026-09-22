import os
import sys
import argparse

import torch
from PIL import Image
from diffusers.models.attention_processor import Attention
from diffusers.image_processor import IPAdapterMaskProcessor

sys.path.append(os.path.dirname(__file__))
from ipattn import get_modules_of_types, insert_monkey, set_ip_adapter_scale_monkey
from main_seg import generate_monkey_image
from image_utils import concat_images_horizontally, concat_images_vertically
from pipelines import CompatibleLatentConsistencyModelPipeline
import datasets

parser = argparse.ArgumentParser()
parser.add_argument("--src_dataset", type=str, default="jlbaker361/league-splash-tagged")
parser.add_argument("--prompt_suffix", type=str, default=" in the jungle", help="appended to the champion name to form the prompt")
parser.add_argument("--dim", type=int, default=256)
parser.add_argument("--initial_steps", type=int, default=4)
parser.add_argument("--final_steps", type=int, default=8)
parser.add_argument("--layer_index", type=int, default=15)
parser.add_argument("--token", type=int, default=1)
parser.add_argument("--kv_type", type=str, default="ip")
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--thresholds", nargs=4, type=float, default=[0.3, 0.5, 0.7, 0.9])
parser.add_argument("--initial_ip_adapter_scales", nargs=4, type=float, default=[0.25, 0.5, 0.75, 1.0])
parser.add_argument("--output_path", type=str, default="threshold_scale_grid.png")


def center_crop_to_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = datasets.load_dataset(args.src_dataset, split="train")
    row = data[0]
    ip_adapter_image = center_crop_to_square(row["image"].convert("RGB")).resize((args.dim, args.dim))
    prompt = row["champion"] + args.prompt_suffix

    pipe = CompatibleLatentConsistencyModelPipeline.from_pretrained(
        "SimianLuo/LCM_Dreamshaper_v7",
        torch_dtype=torch.float16,
    ).to(device)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
    setattr(pipe, "safety_checker", None)

    insert_monkey(pipe)
    attn_list = get_modules_of_types(pipe.unet, Attention)
    mask_processor = IPAdapterMaskProcessor()

    rows = []
    for threshold in args.thresholds:
        row_images = []
        for scale in args.initial_ip_adapter_scales:
            print(f"threshold={threshold} initial_ip_adapter_scale={scale}")
            gen_out = generate_monkey_image(
                pipe, attn_list, mask_processor, ip_adapter_image, prompt,
                dim=args.dim,
                initial_steps=args.initial_steps,
                final_steps=args.final_steps,
                layer_index=args.layer_index,
                token=args.token,
                threshold=threshold,
                kv_type=args.kv_type,
                initial_ip_adapter_scale=scale,
                seed=args.seed,
            )
            row_images.append(gen_out["final_image"])
        rows.append(concat_images_horizontally(row_images))

    grid = concat_images_vertically(rows)
    grid.save(args.output_path)
    print(f"saved {len(args.thresholds)}x{len(args.initial_ip_adapter_scales)} grid to {args.output_path}")


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(args)
