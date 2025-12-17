#what happens when we pass an already existing image to the ip adapter thing

import os
import sys
import argparse
from experiment_helpers.gpu_details import print_details
from experiment_helpers.init_helpers import default_parser,repo_api_init
from accelerate import Accelerator
import time
from diffusers.utils.loading_utils import load_image

import torch.nn.functional as F
import math
from diffusers.models.attention_processor import  IPAdapterAttnProcessor2_0,Attention
from diffusers.image_processor import IPAdapterMaskProcessor
sys.path.append(os.path.dirname(__file__))
from ipattn import MonkeyIPAttnProcessor, get_modules_of_types,reset_monkey,insert_monkey, set_ip_adapter_scale_monkey
import torch
from image_utils import concat_images_horizontally
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from transformers import AutoProcessor, CLIPModel
from pipelines import CompatibleLatentConsistencyModelPipeline
#import ImageReward as RM
from eval_helpers import DinoMetric

parser=default_parser()
parser.add_argument("--layer_index",type=int,default=15)
parser.add_argument("--dim",type=int,default=256)
parser.add_argument("--token",type=int,default=1, help="which IP token is attention")
parser.add_argument("--initial_steps",type=int,default=4,help="how many steps for the initial inference")
parser.add_argument("--initial_mask_step_list",nargs="*",help="steps to generate mask from",type=int)
parser.add_argument("--final_steps",type=int,default=8, help="how many steps for final inference (with mask)")
parser.add_argument("--final_mask_steps_list",nargs="*",help="steps to apply mask from",type=int)
parser.add_argument("--final_adapter_steps_list",nargs="*",help="steps to apply adapter for (regardless of mask)",type=int)
parser.add_argument("--threshold",type=float,default=0.5,help="threshold for mask")

def get_mask(layer_index:int, 
             attn_list:list,step:int,
             token:int,dim:int,
             threshold:float,
             kv_type:str="ip",
             vae_scale:int=8):
    #print("layer",layer_index)
    module=attn_list[layer_index][1] #get the module no name
    #module.processor.kv_ip
    if kv_type=="ip":
        processor_kv=module.processor.kv_ip
    elif kv_type=="str":
        processor_kv=module.processor.kv
    size=processor_kv[step].size()
    #print('\tprocessor_kv[step].size()',processor_kv[step].size())
    
    avg=processor_kv[step].mean(dim=1).squeeze(0)
    #print("\t avg ", avg.size())
    latent_dim=int (math.sqrt(avg.size()[0]))
    #print("\tlatent",latent_dim)
    avg=avg.view([latent_dim,latent_dim,-1])
    #print("\t avg ", avg.size())
    avg=avg[:,:,token]
    #print("\t avg ", avg.size())
    avg_min,avg_max=avg.min(),avg.max()
    x_norm = (avg - avg_min) / (avg_max - avg_min)  # [0,1]
    x_norm[x_norm < threshold]=0.
    avg = (x_norm * 255)
    #avg=F.interpolate(avg.unsqueeze(0).unsqueeze(0), size=(dim, dim), mode="nearest").squeeze(0).squeeze(0)

    return avg

def main(args):
    api,accelerator,device=repo_api_init(args)
    
    pipe = CompatibleLatentConsistencyModelPipeline.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7",
            #torch_dtype=torch.float16,
    ).to(accelerator.device)

    # Load IP-Adapter
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
    set_ip_adapter_scale_monkey(pipe,0.5)

    setattr(pipe,"safety_checker",None)

    insert_monkey(pipe)
    attn_list=get_modules_of_types(pipe.unet,Attention)
    
    generator=torch.Generator()
    generator.manual_seed(123)
    
    ip_adapter_image=load_image("https://assetsio.gnwcdn.com/ASTARION-bg3-crop.jpg?width=1200&height=1200&fit=crop&quality=100&format=png&enable=upscale&auto=webp")
    
    
    initial_image=pipe(" character ",args.dim,args.dim,args.initial_steps,ip_adapter_image=ip_adapter_image,generator=generator).images[0]
    color_rgba = initial_image.convert("RGB")
    for token in [0,1,2,3]:
        mask=sum([get_mask(args.layer_index,attn_list,step,token,args.dim,args.threshold) for step in args.initial_mask_step_list])
        tiny_mask=mask.clone()
        tiny_mask_pil=to_pil_image(1-tiny_mask)
        #print("mask size",mask.size())

        mask=F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(args.dim, args.dim), mode="nearest").squeeze(0).squeeze(0)

        

        mask_pil=to_pil_image(1-mask)
        
        mask_pil = mask_pil.convert("RGB")  # must be single channel for alpha

        #print(mask.size,color_rgba.size)

        # Apply as alpha (translucent mask)
        masked_img=Image.blend(color_rgba, mask_pil, 0.5)
        masked_img.save(f"first_{token}.png")

if __name__=='__main__':
    print_details()
    start=time.time()
    args=parser.parse_args()
    print(args)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")