from diffusers.models.embeddings import IPAdapterFullImageProjection,MultiIPAdapterImageProjection,ImageProjection
from diffusers.models.attention import Attention
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
from diffusers import UNet2DConditionModel
import torch
from typing import List,Union
from diffusers import SanaTransformer2DModel
from diffusers.configuration_utils import ConfigMixin
import json

class MultiIPAdapterImageProjectionWithVisualProjection(torch.nn.Module):
    def __init__(self, multi_ip_adapter:MultiIPAdapterImageProjection,
                 embedding_dim:int, #embedding dim from the embedding model
                 intermediate_embedding_dim:int, #embedding dim that is NOT the cross attention dim or the result of the embedding model
                 device,
                 *args,**kwargs):
        super().__init__(*args,**kwargs)
        self.multi_ip_adapter=multi_ip_adapter.to(device)
        self.visual_projection=torch.nn.Linear(embedding_dim,intermediate_embedding_dim,bias=False).to(device)

    def forward(self,  image_embeds: List[torch.Tensor]):
        image_embeds=[self.visual_projection(image) for image in image_embeds]
        return self.multi_ip_adapter(image_embeds)
    
    def to_json_string(self):
        config_dict={}
        config_dict["_class_name"] = self.__class__.__name__
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"
    
    def __repr__(self):
        return self.__class__.__name__

    
class MultiIPAdapterIdentity(torch.nn.Module,ConfigMixin):
    def __init__(self,num_image_text_embeds:int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_image_text_embeds=num_image_text_embeds

    def forward(self,  image_embeds: List[torch.Tensor]):
        batch_size = image_embeds[0].size()[0]
        image_embeds = [ie.reshape(batch_size, self.num_image_text_embeds, -1) for ie in image_embeds]
        return image_embeds


def get_modules_of_types(model, target_classes):
    return [(name, module) for name, module in model.named_modules()
            if isinstance(module, target_classes)]

def replace_ip_attn(denoising_model:Union[ UNet2DConditionModel,SanaTransformer2DModel]
                    ,embedding_dim:int, #the original embedding dimensions
                    intermediate_embedding_dim:int, #if we have an intermediare global layer
                    cross_attention_dim:int #the input dim for each layer
                    ,num_image_text_embeds:int
                    ,use_projection:bool=True
                    ,use_identity:bool=False,
                    deep_to_ip_layers:bool=False,
                    return_encoder_hid_proj:bool=False):
    layers=get_modules_of_types(denoising_model,IPAdapterAttnProcessor2_0)
    torch_dtype=next(denoising_model.parameters()).dtype
    for (name,module) in layers:
        out_features=module.to_k_ip[0].out_features
        if deep_to_ip_layers:
            average_feature_dim=(cross_attention_dim+out_features)//2
            new_k_ip=torch.nn.ModuleList([
                torch.nn.Sequential(
                torch.nn.Linear(cross_attention_dim,average_feature_dim,bias=True),
                torch.nn.LayerNorm(average_feature_dim),
                torch.nn.Linear(average_feature_dim,out_features,bias=False)) for _ in range(num_image_text_embeds)
            ])
        else:
            new_k_ip=torch.nn.ModuleList([torch.nn.Linear(cross_attention_dim,out_features,bias=False) for _ in range(num_image_text_embeds)])
        new_k_ip.to(denoising_model.device,torch_dtype)
        setattr(module, "to_k_ip",new_k_ip)

        if deep_to_ip_layers:
            average_feature_dim=(cross_attention_dim+out_features)//2
            new_v_ip=torch.nn.ModuleList([
                torch.nn.Sequential(
                torch.nn.Linear(cross_attention_dim,average_feature_dim,bias=True),
                torch.nn.LayerNorm(average_feature_dim),
                torch.nn.Linear(average_feature_dim,out_features,bias=False)) for _ in range(num_image_text_embeds)
            ])
        else:
            new_v_ip=torch.nn.ModuleList([torch.nn.Linear(cross_attention_dim,out_features,bias=False) for _ in range(num_image_text_embeds)])
        new_v_ip.to(denoising_model.device,torch_dtype)
        setattr(module, "to_v_ip",new_v_ip)

    if use_identity:
        multi_ip_adapter=MultiIPAdapterIdentity(num_image_text_embeds)
        if num_image_text_embeds*cross_attention_dim!=embedding_dim:
            print(f"num_image_text_embeds({num_image_text_embeds}(*cross_attention_dim ({cross_attention_dim})!=embedding_dim {embedding_dim}")
    else:

        multi_ip_adapter=MultiIPAdapterImageProjection([ImageProjection(intermediate_embedding_dim,cross_attention_dim,num_image_text_embeds)]).to(device=denoising_model.device)
        #unet.add_module("encoder_hid_proj",multi_ip_adapter)
        if use_projection:
            #unet.encoder_hid_proj=multi_ip_adapter
            multi_ip_adapter=MultiIPAdapterImageProjectionWithVisualProjection(multi_ip_adapter,embedding_dim,intermediate_embedding_dim,denoising_model.device)
    multi_ip_adapter=multi_ip_adapter.to(denoising_model.device,denoising_model.dtype)
    setattr(denoising_model,"encoder_hid_proj",multi_ip_adapter)
    denoising_model.encoder_hid_proj= multi_ip_adapter
    denoising_model.add_module("encoder_hid_proj",multi_ip_adapter)
    #unet.encoder_hid_proj.to(unet.device)
    if return_encoder_hid_proj:
        return multi_ip_adapter

    return denoising_model