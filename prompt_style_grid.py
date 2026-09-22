import os
import sys
import argparse

import torch
from diffusers.models.attention_processor import Attention
from diffusers.image_processor import IPAdapterMaskProcessor

sys.path.append(os.path.dirname(__file__))
from ipattn import get_modules_of_types, insert_monkey
from main_seg import generate_monkey_image
from threshold_scale_grid import center_crop_to_square
from image_utils import concat_images_horizontally, concat_images_vertically
from pipelines import CompatibleLatentConsistencyModelPipeline
import datasets

STYLE_PROMPTS = [" ", ", anime style ", " rennaissance painting", " photorealistic", " childs drawing style ", " comic book style"]

parser = argparse.ArgumentParser()
parser.add_argument("--src_dataset", type=str, default="jlbaker361/league-splash-tagged")
parser.add_argument("--num_images", type=int, default=3)
parser.add_argument("--dim", type=int, default=256)
parser.add_argument("--initial_steps", type=int, default=4)
parser.add_argument("--final_steps", type=int, default=8)
parser.add_argument("--layer_index", type=int, default=15)
parser.add_argument("--token", type=int, default=1)
parser.add_argument("--kv_type", type=str, default="ip")
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--initial_ip_adapter_scale", type=float, default=0.75)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--output_path", type=str, default="prompt_style_grid.png")


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = datasets.load_dataset(args.src_dataset, split="train")

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
    for i in range(args.num_images):
        row = data[i]
        ip_adapter_image = center_crop_to_square(row["image"].convert("RGB")).resize((args.dim, args.dim))

        row_images = []
        for style in STYLE_PROMPTS:
            prompt = row["champion"] + style
            print(f"image={i} champion={row['champion']} prompt={prompt!r}")
            gen_out = generate_monkey_image(
                pipe, attn_list, mask_processor, ip_adapter_image, prompt,
                dim=args.dim,
                initial_steps=args.initial_steps,
                final_steps=args.final_steps,
                layer_index=args.layer_index,
                token=args.token,
                threshold=args.threshold,
                kv_type=args.kv_type,
                initial_ip_adapter_scale=args.initial_ip_adapter_scale,
                seed=args.seed,
            )
            row_images.append(gen_out["final_image"])
        rows.append(concat_images_horizontally(row_images))

    grid = concat_images_vertically(rows)
    grid.save(args.output_path)
    print(f"saved {args.num_images}x{len(STYLE_PROMPTS)} grid to {args.output_path}")


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(args)
