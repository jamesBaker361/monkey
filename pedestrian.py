import kagglehub
import os
import scipy
import cv2 as cv
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
from experiment_helpers.init_helpers import repo_api_init,default_parser,parse_args
from experiment_helpers.gpu_details import print_details
import time
from diffusers.image_processor import VaeImageProcessor
from ipattn import MonkeyIPAttnProcessor, get_modules_of_types,reset_monkey,insert_monkey, set_ip_adapter_scale_monkey,get_mask_rect
import torch
from image_utils import concat_images_horizontally,concat_images_vertically
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from transformers import AutoProcessor, CLIPModel
from pipelines import CompatibleLatentConsistencyModelPipeline,retrieve_timesteps
from diffusers.models.attention_processor import  IPAdapterAttnProcessor2_0,Attention
import torch.nn.functional as F
from scipy.io import loadmat

class PRWDataSet(Dataset):
    
    def __init__(self):
        super().__init__()
        path = kagglehub.dataset_download("edoardomerli/prw-person-re-identification-in-the-wild")
        self.gallery_image_path=[]
        self.query_image_path=[]
        self.bbox=[]
        self.label_list=[]
        print("Path to dataset files:", path)
        print(os.listdir(path))
        frame_mat_file=os.path.join(path,'frame_test.mat')
        frame_mat_content=scipy.io.loadmat(frame_mat_file)
        
        query_box=os.path.join(path,'query_box')
        
        img_index_test=frame_mat_content['img_index_test']
        
        #print(len(img_index_test))
        for j in range(len(img_index_test)):
            target_file=img_index_test[j][0][0]
            #print(target_file)
            target_path=os.path.join(path,"frames",target_file+".jpg")
            for file in os.listdir(os.path.join(path, "annotations")):
                if file.find(target_file)!=-1:
                    label="box_new"
                    #print(os.path.join(path, "annotations",file))
                    try:
                        m=scipy.io.loadmat(os.path.join(path, "annotations",file))[label]
                    except KeyError:
                        label="anno_file"
                        try:
                            m=scipy.io.loadmat(os.path.join(path, "annotations",file))[label]
                        except KeyError:
                            label="anno_previous"
                            try:
                                m=scipy.io.loadmat(os.path.join(path, "annotations",file))[label]
                            except KeyError:
                                print(scipy.io.loadmat(os.path.join(path, "annotations",file)))
                                raise KeyError()
                    for row in m:
                        if row[0]!=-2:
                            [i,x,y,h,w]=row
                            i=str(int(i))
                            for q in os.listdir(query_box):
                                qi=q.split("_")[0]
                                if qi==i:
                                    query_path=os.path.join(query_box,q)
                                    self.gallery_image_path.append(target_path)
                                    self.query_image_path.append(query_path)
                                    self.bbox.append([x,y,h,w])
                                    self.label_list.append(label)
    
    def __len__(self):
        return len(self.gallery_image_path)
    
    
    def __getitem__(self, index):
        return {
            "box":self.bbox[index],
            "gallery": Image.fromarray( cv.imread(self.gallery_image_path[index],cv.COLOR_BGR2RGB)[:, :, ::-1] ),
            "query":Image.fromarray( cv.imread(self.query_image_path[index],cv.COLOR_BGR2RGB)[:, :, ::-1] ),
          #  "label":self.label_list[index]
        }
                        
                        
class CUHKDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.root=os.path.join("cuhk_sysu","cuhk_sysu")
        self.image_path=os.path.join(self.root, "Image","SSM")
        protoc = loadmat(os.path.join(self.root, "annotation","test","train_test","TestG50.mat"))
        protoc = protoc["TestG50"].squeeze()
        
        gallery=protoc["Gallery"][0]
        query=protoc["Query"][0]
        
        self.query_box_list=[]
        self.gallery_box_list=[]
        self.gallery_path_list=[]
        self.query_path_list=[]
        
        for g_list,q in zip(gallery,query):
            #g_list=g_list[0]
            q=q[0]
            query_path=q[0][0]
            query_box=q[1][0]
            query_box[2:]+=query_box[:2]
            for g in g_list:
                gallery_path=g[0][0]
                gallery_box=g[1][0]
                gallery_box[2:]+=gallery_box[:2]
                
                self.query_path_list.append(os.path.join(self.image_path,query_path))
                self.gallery_box_list.append(gallery_box)
                self.query_box_list.append(query_box)
                self.gallery_path_list.append(os.path.join(self.image_path,gallery_path))
                
    def __len__(self):
        return len(self.gallery_path_list)
    
    def __getitem__(self, index):
        return {
            "gallery":Image.fromarray( cv.imread(self.gallery_path_list[index],cv.COLOR_BGR2RGB)[:, :, ::-1] ),
            "query":Image.fromarray( cv.imread(self.query_path_list[index],cv.COLOR_BGR2RGB)[:, :, ::-1] ).crop(self.query_box_list[index]),
            #"query_box":self.query_box_list[index],
            "box":self.gallery_box_list[index]
        }
            
            

def main(args):
    api,accelerator,device=repo_api_init(args)
    dtype={
        "fp16":torch.float16,
        "no":torch.float32,
        "bf16":torch.bfloat16
    }[args.mixed_precision]
    pipe = CompatibleLatentConsistencyModelPipeline.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7",
            dtype=dtype,
    ).to(accelerator.device)
    
    
    

    # Load IP-Adapter
    weight_name={
        "face":"ip-adapter-full-face_sd15.bin",
        "base":"ip-adapter_sd15.bin"
    }[args.ip_weight_name]
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name=weight_name,dtype=dtype)
    

    setattr(pipe,"safety_checker",None)

    insert_monkey(pipe)
    set_ip_adapter_scale_monkey(pipe,0.5)
    attn_list=get_modules_of_types(pipe.unet,Attention)
    reset_monkey(pipe)
    
    generator=torch.Generator()
    generator.manual_seed(123)
    
    timesteps,num_inference_steps=retrieve_timesteps(pipe.scheduler,args.initial_steps)
    print(pipe.scheduler)
    print(timesteps)
    
    for value in ["vae","unet","text_encoder","image_encoder"]:
        if getattr(pipe,value)!=None:
            getattr(pipe,value).to(dtype=dtype)
    
    mask_step_list=[max(m -args.offset,0 )for m in args.mask_step_list]
    if args.dataset.lower()=="cuhk":
        dataset=CUHKDataset()
    else:
        dataset=PRWDataSet()
    
    overlap=[]
    for n, batch in enumerate(dataset):
        if n!=args.limit:
            [x,y,h,w]=batch["box"]
            gallery=batch["gallery"]
            query=batch["query"]
            (img_x,img_y)=gallery.size
            new_x=img_x//args.downscale_factor
            new_y=img_y//args.downscale_factor
            x=x//args.downscale_factor
            y=y//args.downscale_factor
            h=h//args.downscale_factor
            w=w//args.downscale_factor
            gallery=gallery.resize((new_x,new_y))
            
            if args.do_resize:
                (img_x,img_y)=gallery.size
                gallery=gallery.resize((args.resize_dim,args.resize_dim))
                x=int(args.resize_dim * x /img_x)
                y=int(args.resize_dim*y/img_y)
                h=int(args.resize_dim *h/img_x)
                w=int(args.resize_dim *w/img_y)
            
            (img_x,img_y)=gallery.size
            gallery_pt=pipe.image_processor.preprocess(gallery)
            latent_dist=pipe.vae.encode(gallery_pt.to(pipe.vae.device,dtype=dtype)).latent_dist
            noise_level=torch.tensor( timesteps[args.offset]).long()
            latents=latent_dist.sample()
            noisy_latents=pipe.scheduler.add_noise(latents,torch.randn_like(latents),noise_level)
            height=latents.size()[-2]
            width=latents.size()[-1]
            prompt=" "
            pipe(prompt,height,width,args.initial_steps,
                           ip_adapter_image=query,generator=generator,timesteps=timesteps[args.offset:],latents=noisy_latents)
            mask=sum([get_mask_rect(args.layer_index,attn_list,step,args.token,args.threshold) for step in args.mask_step_list])
            mask=F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(img_x,img_y), mode="nearest").squeeze(0).squeeze(0)

            mask_pil=to_pil_image(255-mask)
            draw=ImageDraw.Draw(gallery)
            draw.rectangle([(x,y),(x+h,y+w)],outline="red",width=10)
            concat=concat_images_horizontally([gallery,mask_pil])
            concat.save(f"img_{n}.png")
        else:
            break

if  __name__=='__main__':
    print_details()
    start=time.time()
    parser=default_parser({"repo_id":"jlbaker361/personsearch","project":"person"})
    parser.add_argument("--initial_steps",type=int,default=8)
    parser.add_argument("--token",type=int,default=1)
    parser.add_argument("--layer_index",type=int,default=15)
    parser.add_argument("--offset",type=int,default=2)
    parser.add_argument("--mask_step_list",type=int,nargs="*",default=[2,3,4,5])
    parser.add_argument("--do_resize",action="store_true")
    parser.add_argument("--resize_dim",type=int,default=256)
    parser.add_argument("--downscale_factor",type=int,default=1)
    parser.add_argument("--threshold",type=float,default=0.5,help="threshold for mask")
    parser.add_argument("--ip_weight_name",type=str,default="base",help="base or face")
    parser.add_argument("--dataset",type=str,default="prw",help="one of prw or cuhk")
    args=parse_args(parser)
    print(args)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")