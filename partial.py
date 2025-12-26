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
from image_utils import concat_images_horizontally,concat_images_vertically
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from transformers import AutoProcessor, CLIPModel
from pipelines import CompatibleLatentConsistencyModelPipeline,retrieve_timesteps
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
parser.add_argument("--ip_weight_name",type=str,default="base",help="base or face")

def crop_center_square(img:Image.Image)->Image.Image:
    w, h = img.size
    s = min(w, h)
    return img.crop(((w - s)//2, (h - s)//2, (w + s)//2, (h + s)//2))

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
    
    timesteps,num_inference_steps=retrieve_timesteps(pipe.scheduler,4)
    print(pipe.scheduler)
    print(timesteps)

    # Load IP-Adapter
    weight_name={
        "face":"ip-adapter-full-face_sd15.bin",
        "base":"ip-adapter_sd15.bin"
    }[args.ip_weight_name]
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name=weight_name)
    

    setattr(pipe,"safety_checker",None)

    insert_monkey(pipe)
    set_ip_adapter_scale_monkey(pipe,0.5)
    attn_list=get_modules_of_types(pipe.unet,Attention)
    reset_monkey(pipe)
    
    generator=torch.Generator()
    generator.manual_seed(123)
    
    ip_adapter_image=load_image("https://assetsio.gnwcdn.com/ASTARION-bg3-crop.jpg?width=1200&height=1200&fit=crop&quality=100&format=png&enable=upscale&auto=webp")
    target_image=load_image("https://bg3.wiki/w/images/1/1b/Portrait_Astarion.png")
    target_image=load_image("https://static0.srcdn.com/wordpress/wp-content/uploads/2024/06/baldur-s-gate-3-shadowheart-astarion-karlach.jpg")
    target_image=crop_center_square(target_image)
    target_image_pt=pipe.image_processor.preprocess(target_image,args.dim,args.dim).to(pipe.vae.device)
    latent_dist=pipe.vae.encode(target_image_pt).latent_dist
    
    initial_image=pipe(" on a cobblestone street ",args.dim,args.dim,args.initial_steps,ip_adapter_image=ip_adapter_image,generator=generator).images[0]
    
    initial_image.save("initial.png")
    color_rgba = initial_image.convert("RGB")
    
    
    mask_list_list=[
        [0,1],
        [0],
        [0]
    ]
    offset_list=[
        1,2,3
    ]
    for z,(mask_list,offset) in enumerate( zip(mask_list_list,offset_list)):
        reset_monkey(pipe)
        generator=torch.Generator()
        generator.manual_seed(123)
        noise_level=torch.tensor( timesteps[offset]).long()
        latents=latent_dist.sample()
        noisy_latents=pipe.scheduler.add_noise(latents,torch.randn_like(latents),noise_level)
        initial_image=pipe(" on a cobblestone street ",args.dim,args.dim,args.initial_steps,
                           ip_adapter_image=ip_adapter_image,generator=generator,timesteps=timesteps[offset:],latents=noisy_latents).images[0]
    
        initial_image.save(f"initial_{z}.png")
        vertical_image_list=[]
        for layer in range(len(attn_list)):
            image_list=[]
            try:
                for token in [0,1,2,3]:
                    mask=sum([get_mask(layer,attn_list,step,token,args.dim,args.threshold) for step in mask_list])
                    tiny_mask=mask.clone()
                    tiny_mask_pil=to_pil_image(1-tiny_mask)
                    #print("mask size",mask.size())

                    mask=F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(args.dim, args.dim), mode="nearest").squeeze(0).squeeze(0)

                    mask_pil=to_pil_image(1-mask)
                    
                    mask_pil = mask_pil.convert("RGB")  # must be single channel for alpha

                    #print(mask.size,color_rgba.size)

                    # Apply as alpha (translucent mask)
                    masked_img=Image.blend(target_image.resize((args.dim,args.dim)), mask_pil, 0.5)
                    image_list.append(masked_img)
                vertical_image_list.append(concat_images_horizontally(image_list))
            except Exception as e:
                pass #print("doesnt work for ",layer,e)
        print(len(vertical_image_list),"z= ",z)
        concat_images_vertically(vertical_image_list).save(f"veritcal_{z}.png")

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