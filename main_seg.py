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


parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="seg-ip")
parser.add_argument("--load_hf",action="store_true",help="whether to load a special pretrained model")
parser.add_argument("--embedding",type=str, help="ignore unless load from hf; its the embedding type for embedding helpers")
parser.add_argument("--pretrained_model_path",type=str,default="")
parser.add_argument("--src_dataset",type=str, default="jlbaker361/dreambooth")
parser.add_argument("--use_test_split",action="store_true", help="only true for league dataset")
parser.add_argument("--initial_steps",type=int,default=4,help="how many steps for the initial inference")
parser.add_argument("--initial_mask_step_list",nargs="*",help="steps to generate mask from",type=int)
parser.add_argument("--final_steps",type=int,default=8, help="how many steps for final inference (with mask)")
parser.add_argument("--final_mask_steps_list",nargs="*",help="steps to apply mask from",type=int)
parser.add_argument("--final_adapter_steps_list",nargs="*",help="steps to apply adapter for (regardless of mask)",type=int)
parser.add_argument("--threshold",type=float,default=0.5,help="threshold for mask")
parser.add_argument("--limit",type=int,default=100,help="limit of samples")
parser.add_argument("--layer_index",type=int,default=15)
parser.add_argument("--dim",type=int,default=256)
parser.add_argument("--token",type=int,default=1, help="which IP token is attention")
parser.add_argument("--kv_type",type=str,default="ip")
parser.add_argument("--initial_ip_adapter_scale",type=float,default=0.75)
parser.add_argument("--background",action="store_true")
parser.add_argument("--dest_dataset",type=str, default="jlbaker361/monkey")
parser.add_argument("--object",type=str,default="character")

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

def generate_monkey_image(pipe,
                           attn_list:list,
                           mask_processor:IPAdapterMaskProcessor,
                           ip_adapter_image,
                           prompt:str,
                           dim:int=256,
                           initial_steps:int=4,
                           final_steps:int=8,
                           initial_mask_step_list:list=None,
                           final_mask_steps_list:list=None,
                           final_adapter_steps_list:list=None,
                           layer_index:int=15,
                           token:int=1,
                           threshold:float=0.5,
                           kv_type:str="ip",
                           initial_ip_adapter_scale:float=0.75,
                           background_image=None,
                           seed:int=123) -> dict:
    """
    Runs the core monkey masked-IP-Adapter generation for a single (image,prompt) pair:
    1) a low-scale initial pass to derive the IP-Adapter attention mask,
    2) a final pass that applies that mask over `final_mask_steps_list`.

    Returns a dict with initial_image, mask, mask_pil, masked_img, tiny_mask_pil,
    ip_mask, scale_step_dict, mask_step_list, ip_adapter_image_list and final_image.
    """
    reset_monkey(pipe)

    if initial_mask_step_list is None:
        initial_quarter=initial_steps//4
        initial_mask_step_list=[f for f in range(initial_steps)][initial_quarter:-initial_quarter]
    if final_mask_steps_list is None:
        final_quarter=final_steps//4
        final_mask_steps_list=[f for f in range(final_steps)][final_quarter:-final_quarter]
    if final_adapter_steps_list is None:
        final_adapter_steps_list=final_mask_steps_list

    generator=torch.Generator()
    generator.manual_seed(seed)
    set_ip_adapter_scale_monkey(pipe,initial_ip_adapter_scale)
    initial_image=pipe(prompt,dim,dim,initial_steps,ip_adapter_image=ip_adapter_image,generator=generator).images[0]

    mask=sum([get_mask(layer_index,attn_list,step,token,dim,threshold,kv_type) for step in initial_mask_step_list])
    tiny_mask_pil=to_pil_image(1-mask.clone())

    mask=F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(dim, dim), mode="nearest").squeeze(0).squeeze(0)

    mask_pil=to_pil_image(1-mask)
    color_rgba = initial_image.convert("RGB")
    mask_pil_rgb = mask_pil.convert("RGB")
    masked_img=Image.blend(color_rgba, mask_pil_rgb, 0.5)

    mask[mask>1]=1.
    inverted_mask=1.0-mask

    generator=torch.Generator()
    generator.manual_seed(seed)
    mask_step_list=final_mask_steps_list
    scale_step_dict={i:0 for i in range(final_steps)}
    for k in final_adapter_steps_list:
        scale_step_dict[k]=1.0

    ip_adapter_image_list=ip_adapter_image
    ip_mask=mask_processor.preprocess(mask)
    if background_image is not None:
        ip_adapter_image_list=[[ip_adapter_image, background_image]]
        ip_mask=mask_processor.preprocess([mask,inverted_mask])
        ip_mask=[ip_mask.reshape([1,ip_mask.shape[0],ip_mask.shape[2], ip_mask.shape[3]])]

    final_image=pipe(prompt,dim,dim,final_steps,ip_adapter_image=ip_adapter_image_list,generator=generator,cross_attention_kwargs={
        "ip_adapter_masks":ip_mask
    }, mask_step_list=mask_step_list,scale_step_dict=scale_step_dict).images[0]

    return {
        "initial_image":initial_image,
        "mask":mask,
        "mask_pil":mask_pil,
        "masked_img":masked_img,
        "tiny_mask_pil":tiny_mask_pil,
        "ip_mask":ip_mask,
        "scale_step_dict":scale_step_dict,
        "mask_step_list":mask_step_list,
        "ip_adapter_image_list":ip_adapter_image_list,
        "final_image":final_image,
    }

SUBJECT_PRESERVATION_VARIANTS=["unmasked","raw_mask","normal","all_steps"]
SUBJECT_PRESERVATION_METRIC_NAMES=["subject_preservation","background_divergence","trade_off_ratio","total_lpips"]

class ScoreTracker:
    def __init__(self):
        self.score_list_dict={
                "dino_score_unmasked":[],
                "dino_score_raw_mask":[],
                "dino_score_normal":[],
                "dino_score_all_steps":[],
                "text_score_unmasked":[],
                "text_score_raw_mask":[],
                "text_score_normal":[],
                "text_score_all_steps":[],
                "image_score_unmasked":[],
                "image_score_raw_mask":[],
                "image_score_normal":[],
                "image_score_all_steps":[],
            }
        for variant in SUBJECT_PRESERVATION_VARIANTS:
            for metric_name in SUBJECT_PRESERVATION_METRIC_NAMES:
                self.score_list_dict[f"{metric_name}_{variant}"]=[]

    def update(self,score_dict):
        for k,v in score_dict.items():
            self.score_list_dict[k].append(v)

    def get_means(self)-> dict:
        ret={}
        for k,v in self.score_list_dict.items():
            if len(v)>0:
                ret[k]=np.mean(v)

        return ret

def get_pipe(accelerator:Accelerator,initial_ip_adapter_scale:float)->CompatibleLatentConsistencyModelPipeline:
    pipe = CompatibleLatentConsistencyModelPipeline.from_pretrained(
                "SimianLuo/LCM_Dreamshaper_v7",
                torch_dtype=torch.float16,
    ).to(accelerator.device)

    # Load IP-Adapter
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
    set_ip_adapter_scale_monkey(pipe,initial_ip_adapter_scale)

    setattr(pipe,"safety_checker",None)

    insert_monkey(pipe)
    
    return pipe

def main(args):
    with torch.no_grad():
        #ir_model=RM.load("ImageReward-v1.0")
        
        
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision)
        accelerator.init_trackers(project_name=args.project_name,config=vars(args))

        dino_metric=DinoMetric(accelerator.device)
        subject_preservation_metric=SubjectPreservationMetric(accelerator.device)

        if args.initial_mask_step_list is None:
            initial_quarter=args.initial_steps //4
            args.initial_mask_step_list=[f for f in range(args.initial_steps)][initial_quarter:-initial_quarter]
            accelerator.print("defaulting to initial_mask_step_list",args.initial_mask_step_list )
        if args.final_mask_steps_list is None:
            final_quarter=args.final_steps //4
            args.final_mask_steps_list=[f for f in range(args.final_steps)][final_quarter:-final_quarter]
            accelerator.print("defaulting final maske step lst",args.final_mask_steps_list )
        if args.final_adapter_steps_list is None:
            args.final_adapter_steps_list=args.final_mask_steps_list

        pipe=get_pipe(accelerator,args.initial_ip_adapter_scale)
        
        attn_list=get_modules_of_types(pipe.unet,Attention)

        #monkey_attn_list=get_modules_of_types(pipe.unet,MonkeyIPAttnProcessor)
        try:
            data=datasets.load_dataset(args.src_dataset)
        except:
            data=datasets.load_dataset(args.src_dataset,download_mode="force_redownload")
        data=data["train"]

        

        if args.background:
            background_data=datasets.load_dataset("jlbaker361/real_test_prompt_list",split="train")
            background_dict={row["prompt"]:row["image"] for row in background_data}
            accelerator.print("background dict", background_dict)

        score_tracker=ScoreTracker()
        if args.background:
            background_score_tracker=ScoreTracker()

        output_dict={
        "image":[],
        "augmented_image":[],
        "text_score":[],
        "image_score":[],
        "dino_score":[],
        "prompt":[]
        }

        mask_processor = IPAdapterMaskProcessor()

        for k,row in enumerate(data):
            if k==args.limit:
                break
            ip_adapter_image=row["image"]
            object=row.get("object",row.get("text",args.object))
            prompt=object+real_test_prompt_list[k % len(real_test_prompt_list)]
            if args.background:
                background_image=background_dict[prompt.replace(object,"")]
                prompt=" "

            accelerator.print("generating monkey image")
            gen_out=generate_monkey_image(pipe,attn_list,mask_processor,ip_adapter_image,prompt,
                                           dim=args.dim,
                                           initial_steps=args.initial_steps,
                                           final_steps=args.final_steps,
                                           initial_mask_step_list=args.initial_mask_step_list,
                                           final_mask_steps_list=args.final_mask_steps_list,
                                           final_adapter_steps_list=args.final_adapter_steps_list,
                                           layer_index=args.layer_index,
                                           token=args.token,
                                           threshold=args.threshold,
                                           kv_type=args.kv_type,
                                           initial_ip_adapter_scale=args.initial_ip_adapter_scale,
                                           background_image=background_image if args.background else None)

            initial_image=gen_out["initial_image"]
            mask=gen_out["mask"]
            mask_pil=gen_out["mask_pil"]
            masked_img=gen_out["masked_img"]
            tiny_mask_pil=gen_out["tiny_mask_pil"]
            ip_mask=gen_out["ip_mask"]
            scale_step_dict=gen_out["scale_step_dict"]
            mask_step_list=gen_out["mask_step_list"]
            ip_adapter_image_list=gen_out["ip_adapter_image_list"]
            final_image_raw_mask=gen_out["final_image"]

            masked_list=[]
            for index,[name,module] in enumerate(attn_list):
                if getattr(module,"processor",None)!=None and type(getattr(module,"processor",None))==MonkeyIPAttnProcessor:
                    _mask=sum([get_mask(index,attn_list,step,args.token,args.dim,args.threshold) for step in args.initial_mask_step_list])
                    _mask=F.interpolate(_mask.unsqueeze(0).unsqueeze(0), size=(args.dim, args.dim), mode="nearest").squeeze(0).squeeze(0)

                

                    ''''bw_img = Image.fromarray(_mask.cpu().numpy(), mode="L")  # "L" = 8-bit grayscale
                    _mask_pil = ImageOps.invert(bw_img)'''
                    color_rgba = initial_image.convert("RGB")
                    _mask_pil = to_pil_image(1-_mask).convert("RGB")  # must be single channel for alpha

                    #print(_mask.size(),_mask_pil.size,color_rgba.size)

                    # Apply as alpha (translucent mask)
                    _masked_img=Image.blend(color_rgba, _mask_pil, 0.5)

                    masked_list.append(_masked_img)

            first_concat=concat_images_horizontally(masked_list)

            accelerator.log({
                "first_concat":wandb.Image(first_concat)
            })

            accelerator.print("mask step list",mask_step_list)
            accelerator.print("scale step dict",scale_step_dict)

            generator=torch.Generator()
            generator.manual_seed(123)
            set_ip_adapter_scale_monkey(pipe,1.0)
            accelerator.print("final image unmasked")
            final_image_unmasked=pipe(prompt,args.dim,args.dim,args.final_steps,ip_adapter_image=ip_adapter_image_list,generator=generator,
                                      scale_step_dict=scale_step_dict).images[0]
            torch.cuda.empty_cache()

            generator=torch.Generator()
            generator.manual_seed(123)
            set_ip_adapter_scale_monkey(pipe,1.0)
            accelerator.print("final_image_normal")
            final_image_normal=pipe(prompt,args.dim,args.dim,args.final_steps,ip_adapter_image=ip_adapter_image_list,generator=generator).images[0]
            torch.cuda.empty_cache()

            generator=torch.Generator()
            generator.manual_seed(123)
            set_ip_adapter_scale_monkey(pipe,1.0)
            accelerator.print("final_image_all_steps")
            final_image_all_steps=final_image_raw_mask=pipe(prompt,args.dim,args.dim,args.final_steps,ip_adapter_image=ip_adapter_image_list,generator=generator,cross_attention_kwargs={
                "ip_adapter_masks":ip_mask
            }, mask_step_list=[x for x in range(args.final_steps)],scale_step_dict={i:1.0  for i in range(args.final_steps) }).images[0]
            accelerator.print("all steps ",[x for x in range(args.final_steps)],{i:1.0  for i in range(args.final_steps) })
            mask=mask.cpu()

            concat_image_list=[ip_adapter_image.resize([args.dim,args.dim],0),mask_pil,masked_img,
                                               initial_image,
                                               final_image_raw_mask,
                                               final_image_unmasked,
                                               final_image_normal,
                                               final_image_all_steps]
            if args.background:
                concat_image_list=[background_image]+concat_image_list
            concat=concat_images_horizontally(concat_image_list)
            accelerator.log({
                "image": wandb.Image(concat)
            })

            

            accelerator.log({"tiny_mask":wandb.Image(tiny_mask_pil)})


            inputs = processor(
                text=[prompt], images=[ip_adapter_image,final_image_normal,final_image_unmasked,final_image_raw_mask,final_image_all_steps], return_tensors="pt", padding=True
            )

            outputs = clip_model(**inputs)

            #logits_per_text = outputs.logits_per_text.numpy()[0]  # this is the image-text similarity score
            image_embeds=outputs.image_embeds
            text_embeds=outputs.text_embeds
            logits_per_text=torch.matmul(text_embeds, image_embeds.t())[0]
            #accelerator.print("logits",logits_per_text.size())

            image_similarities=torch.matmul(image_embeds,image_embeds.t()).numpy()[0]
            [_,text_score_normal,text_score_unmasked, text_score_raw_mask,text_score_all_steps]=logits_per_text
            [_,image_score_normal,image_score_unmasked, image_score_raw_mask,image_score_all_steps]=image_similarities
            #[ir_score_normal,ir_score_unmasked, ir_score_raw_mask,ir_score_all_steps]=ir_model.score(prompt,[final_image_normal,final_image_unmasked,final_image_raw_mask,final_image_all_steps])
            [dino_score_normal,dino_score_unmasked, dino_score_raw_mask,dino_score_all_steps]=dino_metric.get_scores(ip_adapter_image, [final_image_normal,final_image_unmasked,final_image_raw_mask,final_image_all_steps])

            original_tensor=to_tensor(ip_adapter_image.convert("RGB").resize((args.dim,args.dim)))
            variant_image_mask_dict={
                "normal":(final_image_normal,mask),
                "unmasked":(final_image_unmasked,mask),
                "raw_mask":(final_image_raw_mask,mask),
                "all_steps":(final_image_all_steps,mask),
            }
            subject_score_dict={}
            for variant,(generated_image,subject_mask) in variant_image_mask_dict.items():
                generated_tensor=to_tensor(generated_image.convert("RGB").resize((args.dim,args.dim)))
                subject_mask_tensor=subject_mask.float().unsqueeze(0)
                preservation_scores=subject_preservation_metric.compute_preservation_score(
                    original_tensor,generated_tensor,subject_mask_tensor
                )
                for metric_name,value in preservation_scores.items():
                    subject_score_dict[f"{metric_name}_{variant}"]=value
            accelerator.print(subject_score_dict)
            accelerator.log(subject_score_dict)
            score_tracker.update(subject_score_dict)



            score_dict={
                "dino_score_unmasked":dino_score_unmasked,
                "dino_score_raw_mask":dino_score_raw_mask,
                "dino_score_normal":dino_score_normal,
                "dino_score_all_steps":dino_score_all_steps,
                "text_score_unmasked":text_score_unmasked,
                "text_score_raw_mask":text_score_raw_mask,
                "text_score_normal":text_score_normal,
                "text_score_all_steps":text_score_all_steps,
                "image_score_unmasked":image_score_unmasked,
                "image_score_raw_mask":image_score_raw_mask,
                "image_score_normal":image_score_normal,
                "image_score_all_steps":image_score_all_steps
            }
            accelerator.print(score_dict)
            accelerator.log(score_dict)

            score_tracker.update(score_dict)

            output_dict["augmented_image"].append(final_image_raw_mask)
            output_dict["image"].append(ip_adapter_image)
            output_dict["dino_score"].append(dino_score_raw_mask)
            output_dict["image_score"].append(image_score_raw_mask)
            output_dict["text_score"].append(text_score_raw_mask)
            output_dict["prompt"].append(prompt)

            if args.background:
                inputs = processor(
                text=[prompt], images=[background_image,final_image_normal,final_image_unmasked,final_image_raw_mask,final_image_all_steps], return_tensors="pt", padding=True
                )
                outputs = clip_model(**inputs)

                image_embeds=outputs.image_embeds
                text_embeds=outputs.text_embeds
                logits_per_text=torch.matmul(text_embeds, image_embeds.t())[0]

                image_similarities=torch.matmul(image_embeds,image_embeds.t()).numpy()[0]
                [_,text_score_normal,text_score_unmasked, text_score_raw_mask,text_score_all_steps]=logits_per_text
                [_,image_score_normal,image_score_unmasked, image_score_raw_mask,image_score_all_steps]=image_similarities
                #[ir_score_normal,ir_score_unmasked, ir_score_raw_mask,ir_score_all_steps]=ir_model.score(prompt,[final_image_normal,final_image_unmasked,final_image_raw_mask,final_image_all_steps])




                score_dict={
                    "image_score_unmasked":image_score_unmasked,
                    "image_score_raw_mask":image_score_raw_mask,
                    "image_score_normal":image_score_normal,
                    "image_score_all_steps":image_score_all_steps
                }

                for k,v in score_dict.items():
                    accelerator.print("background_"+k,v)

                background_score_tracker.update(score_dict)

        avg_score_dict=score_tracker.get_means()

        Dataset.from_dict(output_dict).push_to_hub(args.dest_dataset)
        accelerator.print("Average Scores:")
        accelerator.print(len(avg_score_dict))
        for k,v in avg_score_dict.items():
            accelerator.print(k,float(v))
        if args.background:
            avg_score_dict=background_score_tracker.get_means()

            accelerator.print("Background Average Scores:")
            accelerator.print(len(avg_score_dict))
            for k,v in avg_score_dict.items():
                accelerator.print(k,float(v))





        




    return

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