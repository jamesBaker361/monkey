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
from torchvision.transforms.functional import to_pil_image
from transformers import AutoProcessor, CLIPModel
from pipelines import CompatibleLatentConsistencyModelPipeline
#import ImageReward as RM
from eval_helpers import DinoMetric


#from controlnet_aux import HEDdetector, MidasDetector, MLSDdetector, OpenposeDetector, PidiNetDetector, NormalBaeDetector, LineartDetector, LineartAnimeDetector, CannyDetector, ContentShuffleDetector, ZoeDetector, MediapipeFaceDetector, SamDetector, LeresDetector, DWposeDetector
from custom_sam_detector import CustomSamDetector
import datasets
from datasets import Dataset
import wandb
import numpy as np
from prompt_list import real_test_prompt_list

parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="seg-ip")
parser.add_argument("--load_hf",action="store_true",help="whether to load a special pretrained model")
parser.add_argument("--embedding",type=str, help="ignore unless load from hf; its the embedding type for embedding helpers")
parser.add_argument("--pretrained_model_path",type=str,default="")
parser.add_argument("--src_dataset",type=str, default="jlbaker361/ssl-league_captioned_splash-1000-sana")
parser.add_argument("--use_test_split",action="store_true", help="only true for league dataset")
parser.add_argument("--initial_steps",type=int,default=4,help="how many steps for the initial inference")
parser.add_argument("--initial_mask_step_list",nargs="*",help="steps to generate mask from",type=int)
parser.add_argument("--final_steps",type=int,default=8, help="how many steps for final inference (with mask)")
parser.add_argument("--final_mask_steps_list",nargs="*",help="steps to apply mask from",type=int)
parser.add_argument("--final_adapter_steps_list",nargs="*",help="steps to apply adapter for (regardless of mask)",type=int)
parser.add_argument("--threshold",type=float,default=0.5,help="threshold for mask")
parser.add_argument("--limit",type=int,default=-1,help="limit of samples")
parser.add_argument("--layer_index",type=int,default=15)
parser.add_argument("--dim",type=int,default=256)
parser.add_argument("--token",type=int,default=1, help="which IP token is attention")
parser.add_argument("--overlap_frac",type=float,default=0.8)
parser.add_argument("--segmentation_attention_method",type=str,help="overlap or exclusive",default="overlap")
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

class ScoreTracker:
    def __init__(self):
        self.score_list_dict={
                "dino_score_unmasked":[],
                "dino_score_seg_mask":[],
                "dino_score_raw_mask":[],
                "dino_score_normal":[],
                "dino_score_all_steps":[],
                "text_score_unmasked":[],
                "text_score_seg_mask":[],
                "text_score_raw_mask":[],
                "text_score_normal":[],
                "text_score_all_steps":[],
                "image_score_unmasked":[],
                "image_score_seg_mask":[],
                "image_score_raw_mask":[],
                "image_score_normal":[],
                "image_score_all_steps":[],
            }

    def update(self,score_dict):
        for k,v in score_dict.items():
            self.score_list_dict[k].append(v)

    def get_means(self)-> dict:
        ret={}
        for k,v in self.score_list_dict.items():
            if len(v)>0:
                ret[k]=np.mean(v)

        return ret

def main(args):
    with torch.no_grad():
        #ir_model=RM.load("ImageReward-v1.0")
        
        
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision)
        accelerator.init_trackers(project_name=args.project_name,config=vars(args))

        dino_metric=DinoMetric(accelerator.device)

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

        custom_sam= CustomSamDetector.from_pretrained("ybelkada/segment-anything", subfolder="checkpoints").to(accelerator.device)

        pipe = CompatibleLatentConsistencyModelPipeline.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7",
            torch_dtype=torch.float16,
        ).to(accelerator.device)

        # Load IP-Adapter
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
        set_ip_adapter_scale_monkey(pipe,args.initial_ip_adapter_scale)

        setattr(pipe,"safety_checker",None)

        insert_monkey(pipe)
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

        for k,row in enumerate(data):
            if k==args.limit:
                break
            reset_monkey(pipe)
            ip_adapter_image=row["image"]
            object=args.object
            if "object" in row:
                object=row["object"]
            prompt=object+real_test_prompt_list[k % len(real_test_prompt_list)]
            if args.background:
                background_image=background_dict[prompt.replace(object,"")]
                prompt=" "
            generator=torch.Generator()
            generator.manual_seed(123)
            set_ip_adapter_scale_monkey(pipe,0.5)
            accelerator.print("inital image")
            initial_image=pipe(prompt,args.dim,args.dim,args.initial_steps,ip_adapter_image=ip_adapter_image,generator=generator).images[0]

            mask=sum([get_mask(args.layer_index,attn_list,step,args.token,args.dim,args.threshold) for step in args.initial_mask_step_list])
            tiny_mask=mask.clone()
            tiny_mask_pil=to_pil_image(1-tiny_mask)
            #print("mask size",mask.size())

            mask=F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(args.dim, args.dim), mode="nearest").squeeze(0).squeeze(0)

            

            mask_pil=to_pil_image(1-mask)
            color_rgba = initial_image.convert("RGB")
            mask_pil = mask_pil.convert("RGB")  # must be single channel for alpha

            #print(mask.size,color_rgba.size)

            # Apply as alpha (translucent mask)
            masked_img=Image.blend(color_rgba, mask_pil, 0.5)

            mask[mask>1]=1.
            inverted_mask=1.0-mask
            #mask=1-mask
            #print(mask.size())
            mask_processor = IPAdapterMaskProcessor()
            #print(mask_processor.config)
            '''mask = mask_processor.preprocess(mask)
            inverted_mask=mask_processor.preprocess(inverted_mask)'''
            #print(mask.size())
            #print("mask size",mask.size())

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

            generator=torch.Generator()
            generator.manual_seed(123)
            mask_step_list=args.final_mask_steps_list        
            scale_step_dict={i:0  for i in range(args.final_steps) }
            accelerator.print("mask step list",mask_step_list)
            
            for k in args.final_adapter_steps_list:
                scale_step_dict[k]=1.0
            accelerator.print("scale step dict",scale_step_dict)
            ip_adapter_image_list=ip_adapter_image
            ip_mask=mask_processor.preprocess(mask)
            if args.background:
                ip_adapter_image_list=[[ip_adapter_image, background_image]]
                ip_mask=mask_processor.preprocess([mask,inverted_mask])
                accelerator.print('ip_mask.size()',ip_mask.size())
                ip_mask=[ip_mask.reshape([1,ip_mask.shape[0],ip_mask.shape[2], ip_mask.shape[3]])]
                
            accelerator.print("final image raw mask")
            final_image_raw_mask=pipe(prompt,args.dim,args.dim,args.final_steps,ip_adapter_image=ip_adapter_image_list,generator=generator,cross_attention_kwargs={
                "ip_adapter_masks":ip_mask
            }, mask_step_list=mask_step_list,scale_step_dict=scale_step_dict).images[0]

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
            segmented_image,map_list=custom_sam(initial_image,detect_resolution=args.dim)
            accelerator.log({
                "segmented":wandb.Image(segmented_image)
            })
            mask=mask.cpu()

            if args.segmentation_attention_method=="exclusive":
                map_mask=torch.ones((args.dim,args.dim))
            elif args.segmentation_attention_method=="overlap":
                map_mask=torch.zeros((args.dim,args.dim))
            for ann in map_list:
                map_=ann["segmentation"]

                map_=torch.from_numpy(map_).cpu()
                #map_=F.interpolate(map_.unsqueeze(0).unsqueeze(0), (args.dim,args.dim)).squeeze(0).squeeze(0)
                
                if args.segmentation_attention_method=="exclusive":
                    merged=map_*mask

                    map_mask=merged*map_mask

                elif args.segmentation_attention_method=="overlap":
                    
                    n_ones=map_.sum()
                    merged=map_*mask
                    if merged.sum()>= args.overlap_frac * n_ones:
                        map_mask=torch.max(map_,map_mask)
                
            if len(map_list)==0:
                accelerator.log({
                    "unsegmentable":wandb.Image(initial_image)
                })
            for _ in range(2):
                if len(map_mask.size())>2:
                    map_mask=map_mask.squeeze(0)

            
            inverted_map_mask=1.0-map_mask
            map_mask_pil=to_pil_image(1-map_mask).convert("RGB")
            #map_mask=mask_processor.preprocess(map_mask)

            generator=torch.Generator()
            generator.manual_seed(123)
            ip_map_mask=mask_processor.preprocess(map_mask)
            if args.background:
                ip_map_mask=mask_processor.preprocess([map_mask, inverted_map_mask])
                accelerator.print('ip_map_mask.size()',ip_map_mask.size())
                ip_map_mask = [ip_map_mask.reshape(1, ip_map_mask.shape[0], ip_map_mask.shape[2], ip_map_mask.shape[3])]
                
            accelerator.print("final_image_seg_mask")
            final_image_seg_mask=pipe(prompt,args.dim,args.dim,args.final_steps,ip_adapter_image=ip_adapter_image_list,generator=generator,cross_attention_kwargs={
                "ip_adapter_masks":ip_map_mask
            }, mask_step_list=mask_step_list,scale_step_dict=scale_step_dict).images[0]

            
            concat_image_list=[ip_adapter_image.resize([args.dim,args.dim],0),mask_pil,map_mask_pil,masked_img, segmented_image,
                                               initial_image,
                                               final_image_raw_mask,
                                               final_image_seg_mask,
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
                text=[prompt], images=[ip_adapter_image,final_image_normal,final_image_unmasked,final_image_seg_mask,final_image_raw_mask,final_image_all_steps], return_tensors="pt", padding=True
            )

            outputs = clip_model(**inputs)
            
            #logits_per_text = outputs.logits_per_text.numpy()[0]  # this is the image-text similarity score
            image_embeds=outputs.image_embeds
            text_embeds=outputs.text_embeds
            logits_per_text=torch.matmul(text_embeds, image_embeds.t())[0]
            #accelerator.print("logits",logits_per_text.size())

            image_similarities=torch.matmul(image_embeds,image_embeds.t()).numpy()[0]
            [_,text_score_normal,text_score_unmasked, text_score_seg_mask, text_score_raw_mask,text_score_all_steps]=logits_per_text
            [_,image_score_normal,image_score_unmasked, image_score_seg_mask, image_score_raw_mask,image_score_all_steps]=image_similarities
            #[ir_score_normal,ir_score_unmasked, ir_score_seg_mask, ir_score_raw_mask,ir_score_all_steps]=ir_model.score(prompt,[final_image_normal,final_image_unmasked,final_image_seg_mask,final_image_raw_mask,final_image_all_steps])
            [dino_score_normal,dino_score_unmasked, dino_score_seg_mask, dino_score_raw_mask,dino_score_all_steps]=dino_metric.get_scores(ip_adapter_image, [final_image_normal,final_image_unmasked,final_image_seg_mask,final_image_raw_mask,final_image_all_steps])

            

            score_dict={
                "dino_score_unmasked":dino_score_unmasked,
                "dino_score_seg_mask":dino_score_seg_mask,
                "dino_score_raw_mask":dino_score_raw_mask,
                "dino_score_normal":dino_score_normal,
                "dino_score_all_steps":dino_score_all_steps,
                "text_score_unmasked":text_score_unmasked,
                "text_score_seg_mask":text_score_seg_mask,
                "text_score_raw_mask":text_score_raw_mask,
                "text_score_normal":text_score_normal,
                "text_score_all_steps":text_score_all_steps,
                "image_score_unmasked":image_score_unmasked,
                "image_score_seg_mask":image_score_seg_mask,
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
                text=[prompt], images=[background_image,final_image_normal,final_image_unmasked,final_image_seg_mask,final_image_raw_mask,final_image_all_steps], return_tensors="pt", padding=True
                )
                outputs = clip_model(**inputs)

                image_embeds=outputs.image_embeds
                text_embeds=outputs.text_embeds
                logits_per_text=torch.matmul(text_embeds, image_embeds.t())[0]

                image_similarities=torch.matmul(image_embeds,image_embeds.t()).numpy()[0]
                [_,text_score_normal,text_score_unmasked, text_score_seg_mask, text_score_raw_mask,text_score_all_steps]=logits_per_text
                [_,image_score_normal,image_score_unmasked, image_score_seg_mask, image_score_raw_mask,image_score_all_steps]=image_similarities
                #[ir_score_normal,ir_score_unmasked, ir_score_seg_mask, ir_score_raw_mask,ir_score_all_steps]=ir_model.score(prompt,[final_image_normal,final_image_unmasked,final_image_seg_mask,final_image_raw_mask,final_image_all_steps])


                

                score_dict={
                    "image_score_unmasked":image_score_unmasked,
                    "image_score_seg_mask":image_score_seg_mask,
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