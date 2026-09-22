import os
import sys
import argparse
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
import time

import torch.nn.functional as F
import math
from diffusers.models.attention_processor import  IPAdapterAttnProcessor2_0,Attention
from diffusers.image_processor import IPAdapterMaskProcessor
sys.path.append(os.path.dirname(__file__))
from ipattn import MonkeyIPAttnProcessor, get_modules_of_types,reset_monkey,insert_monkey, set_ip_adapter_scale_monkey
import torch
from image_utils import concat_images_horizontally
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor
from transformers import AutoProcessor, CLIPModel
from pipelines import CompatibleLatentConsistencyModelPipeline
#import ImageReward as RM
from eval_helpers import DinoMetric, SubjectPreservationMetric


#from controlnet_aux import HEDdetector, MidasDetector, MLSDdetector, OpenposeDetector, PidiNetDetector, NormalBaeDetector, LineartDetector, LineartAnimeDetector, CannyDetector, ContentShuffleDetector, ZoeDetector, MediapipeFaceDetector, SamDetector, LeresDetector, DWposeDetector
import datasets
from datasets import Dataset
import wandb
import numpy as np
import pandas as pd
from prompt_list import real_test_prompt_list
import matplotlib.pyplot as plt
import seaborn as sns



def get_mask(layer_index:int,
             attn_list:list,step:int,
             token:int,dim:int,
             threshold:float,
             kv_type:str="ip",
             vae_scale:int=8):
    module=attn_list[layer_index][1] #get the module no name
    if kv_type=="ip":
        processor_kv=module.processor.kv_ip
    elif kv_type=="str":
        processor_kv=module.processor.kv

    avg=processor_kv[step].mean(dim=1).squeeze(0)
    latent_dim=int(math.sqrt(avg.size()[0]))
    avg=avg.view([latent_dim,latent_dim,-1])
    avg=avg[:,:,token]
    avg_min,avg_max=avg.min(),avg.max()
    x_norm = (avg - avg_min) / (avg_max - avg_min)  # [0,1]
    x_norm[x_norm < threshold]=0.
    avg = (x_norm * 255)

    return avg


def _quarter_trim_step_list(step_count:int)->list:
    step_list=[f for f in range(step_count)]
    quarter=step_count//4
    if quarter>0:
        step_list=step_list[quarter:-quarter]
    return step_list


def run_sensitivity_analysis(pipe, sample_image, sample_prompt,
                             steps_range=[2, 4, 6, 8],
                             thresholds=[0.3, 0.5, 0.7, 0.9],
                             output_dir="sensitivity_analysis",
                             initial_steps=4,
                             initial_ip_adapter_scale=0.75,
                             layer_index=15,
                             token=1,
                             dim=256,
                             kv_type="ip"):
    """
    Measure how text alignment varies with key hyperparameters, using the
    same monkey-patched, attention-mask-guided IP-Adapter generation as
    main_seg.py (insert_monkey + get_mask) instead of vanilla IP-Adapter
    generation.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    mask_processor = IPAdapterMaskProcessor()

    insert_monkey(pipe)
    attn_list=get_modules_of_types(pipe.unet,Attention)

    results = []

    for steps in steps_range:
        reset_monkey(pipe)

        # initial low-scale pass to record the IP-Adapter attention maps used for masking
        generator = torch.Generator()
        generator.manual_seed(42)
        set_ip_adapter_scale_monkey(pipe, initial_ip_adapter_scale)
        pipe(
            sample_prompt,
            dim, dim,
            initial_steps,
            ip_adapter_image=sample_image,
            generator=generator
        )

        initial_mask_step_list=_quarter_trim_step_list(initial_steps)
        final_mask_step_list=_quarter_trim_step_list(steps)
        scale_step_dict={i:0 for i in range(steps)}
        for i in final_mask_step_list:
            scale_step_dict[i]=1.0

        for threshold in thresholds:
            mask=sum([get_mask(layer_index,attn_list,step,token,dim,threshold,kv_type) for step in initial_mask_step_list])
            mask=F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(dim, dim), mode="nearest").squeeze(0).squeeze(0)
            mask[mask>1]=1.
            ip_mask=mask_processor.preprocess(mask)

            # masked final pass
            generator = torch.Generator()
            generator.manual_seed(42)
            set_ip_adapter_scale_monkey(pipe, 1.0)

            image = pipe(
                sample_prompt,
                dim, dim,
                steps,
                ip_adapter_image=sample_image,
                generator=generator,
                cross_attention_kwargs={
                    "ip_adapter_masks":ip_mask
                },
                mask_step_list=final_mask_step_list,
                scale_step_dict=scale_step_dict
            ).images[0]

            # Score with CLIP
            inputs = processor(text=[sample_prompt], images=[image],
                             return_tensors="pt", padding=True)
            outputs = clip_model(**inputs)
            text_score = outputs.logits_per_text[0, 0].item()

            results.append({
                'steps': steps,
                'threshold': threshold,
                'text_score': text_score,
            })

            print(f"Steps={steps}, Threshold={threshold}: Score={text_score:.3f}")

    # Create heatmap
    df = pd.DataFrame(results)
    pivot = df.pivot(index='steps', columns='threshold', values='text_score')

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.2, vmax=0.8)
    plt.title('Sensitivity Analysis: Inference Steps vs Threshold')
    plt.ylabel('Inference Steps')
    plt.xlabel('Threshold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sensitivity_heatmap.png", dpi=150)

    print(f"Sensitivity analysis saved to {output_dir}/")
    return df


if __name__=='__main__':
    print_details()
    start=time.time()

    parser=argparse.ArgumentParser()
    parser.add_argument("--mixed_precision",type=str,default="no")
    parser.add_argument("--src_dataset",type=str, default="jlbaker361/ssl-league_captioned_splash-1000-sana")
    parser.add_argument("--sample_index",type=int,default=0,help="which row of src_dataset to use as the sample image")
    parser.add_argument("--object",type=str,default="character")
    parser.add_argument("--sample_prompt",type=str,default=None,help="overrides the prompt built from --object and the dataset row")
    parser.add_argument("--steps_range",nargs="*",type=int,default=[2, 4, 6, 8])
    parser.add_argument("--thresholds",nargs="*",type=float,default=[0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--output_dir",type=str,default="sensitivity_analysis")
    parser.add_argument("--initial_steps",type=int,default=4,help="how many steps for the initial (mask-generating) pass")
    parser.add_argument("--initial_ip_adapter_scale",type=float,default=0.75)
    parser.add_argument("--layer_index",type=int,default=15)
    parser.add_argument("--token",type=int,default=1, help="which IP token is attention")
    parser.add_argument("--dim",type=int,default=256)
    parser.add_argument("--kv_type",type=str,default="ip")
    args=parser.parse_args()
    print(args)

    accelerator=Accelerator(mixed_precision=args.mixed_precision)

    pipe = CompatibleLatentConsistencyModelPipeline.from_pretrained(
        "SimianLuo/LCM_Dreamshaper_v7",
        torch_dtype=torch.float16,
    ).to(accelerator.device)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
    setattr(pipe,"safety_checker",None)

    try:
        data=datasets.load_dataset(args.src_dataset)
    except:
        data=datasets.load_dataset(args.src_dataset,download_mode="force_redownload")
    data=data["train"]

    row=data[args.sample_index]
    sample_image=row["image"]
    object=args.object
    if "object" in row:
        object=row["object"]
    sample_prompt=args.sample_prompt or object+real_test_prompt_list[args.sample_index % len(real_test_prompt_list)]
    print("sample_prompt",sample_prompt)

    run_sensitivity_analysis(pipe, sample_image, sample_prompt,
                             steps_range=args.steps_range,
                             thresholds=args.thresholds,
                             output_dir=args.output_dir,
                             initial_steps=args.initial_steps,
                             initial_ip_adapter_scale=args.initial_ip_adapter_scale,
                             layer_index=args.layer_index,
                             token=args.token,
                             dim=args.dim,
                             kv_type=args.kv_type)

    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")

