import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import gc
#from controlnet_aux import SamDetector
import os
import sys
sys.path.append(os.path.dirname(__file__))
from custom_sam_detector import CustomSamDetector


import torch
from PIL import Image
from diffusers.loaders.ip_adapter import IPAdapterMixin

# Load human segmentation preprocessor
#sam =  SamDetector.from_pretrained("ybelkada/segment-anything", subfolder="checkpoints")




'''gen=torch.Generator()
gen.manual_seed(123)
gen_image=pipe("cat",height=dim,width=dim,num_inference_steps=4,ip_adapter_image=ip_adapter_image,generator=gen)

gen_image.images[0]

[(n[0],n[1].__class__) for n in pipe.unet.named_modules() if n[0].endswith("processor")]

pipe.unet.mid_block.attentions[0].transformer_blocks[0].attn2.processor.scale'''

from diffusers.models.attention_processor import  IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0, IPAdapterXFormersAttnProcessor,Attention
from diffusers.utils import deprecate, is_torch_xla_available, logging
from typing import Optional,List
from diffusers.image_processor import IPAdapterMaskProcessor

from adapter_helpers import get_modules_of_types
import math
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from diffusers.loaders.unet_loader_utils import _maybe_expand_lora_scales




#sam =  SamDetector.from_pretrained("ybelkada/segment-anything", subfolder="checkpoints")

big_global_dict={}
big_global_ip_dict={}

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def add_padding_with_text(img: Image.Image, text: str, pad_width: int = 100, font_path=None, font_size=24):
    """
    Add white padding to the left of an image and draw text in that space.

    Args:
        img (PIL.Image): Input image.
        text (str): Text to write in the padding area.
        pad_width (int): Width of the white padding to add on the left.
        font_path (str): Optional path to a .ttf font file.
        font_size (int): Font size for the text.
    """
    w, h = img.size
    
    # Create new white canvas with extra width
    new_img = Image.new("RGB", (w + pad_width, h), "white")

    # Paste original image on the right
    new_img.paste(img, (pad_width, 0))

    # Prepare to draw
    draw = ImageDraw.Draw(new_img)
    
    # Load font (default to PIL built-in if no path given)
    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()

    

    # Draw text in black
    draw.text((h//8, h//2), text, fill="black", font_size=font_size)

    return new_img

class MonkeyIPAttnProcessor(torch.nn.Module):
    def __init__(self,processor:IPAdapterAttnProcessor2_0,dict_name:str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.processor=processor
        self.scale=processor.scale
        self.dict_name=dict_name
        self.kv=[]
        self.kv_ip=[]

    def reset(self):
        self.kv.clear()
        self.kv_ip.clear()
        gc.collect()
        torch.cuda.empty_cache()

    def __call__(self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        ip_adapter_masks: Optional[torch.Tensor] = None,
    ):
        #print("h")
        residual = hidden_states

        # separate ip_hidden_states from encoder_hidden_states
        if encoder_hidden_states is not None:
            if isinstance(encoder_hidden_states, tuple):
                encoder_hidden_states, ip_hidden_states = encoder_hidden_states
            else:
                deprecation_message = (
                    "You have passed a tensor as `encoder_hidden_states`. This is deprecated and will be removed in a future release."
                    " Please make sure to update your script to pass `encoder_hidden_states` as a tuple to suppress this warning."
                )
                deprecate("encoder_hidden_states not a tuple", "1.0.0", deprecation_message, standard_warn=False)
                end_pos = encoder_hidden_states.shape[1] - self.processor.num_tokens[0]
                encoder_hidden_states, ip_hidden_states = (
                    encoder_hidden_states[:, :end_pos, :],
                    [encoder_hidden_states[:, end_pos:, :]],
                )

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        print("hidden states shape",hidden_states.size())

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        #print("\t hidden states shape",hidden_states.size())
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        #print("\t query size",query.size())

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        #print("\t k size",key.size())
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        #print("\t query size",query.size())

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        #print("\t k size",key.size())
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        #print("\t hidden states shape after scaled dot product",hidden_states.size())
        attn_weight = query @ key.transpose(-2, -1)
        attn_weight = torch.softmax(attn_weight, dim=-1)

        #print("\t attn_weight shape",attn_weight.size())

        self.kv.append(attn_weight)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        #print("\t hidden states shape after transpose",hidden_states.size())
        hidden_states = hidden_states.to(query.dtype)

        if ip_adapter_masks is not None:
            if not isinstance(ip_adapter_masks, List):
                # for backward compatibility, we accept `ip_adapter_mask` as a tensor of shape [num_ip_adapter, 1, height, width]
                ip_adapter_masks = list(ip_adapter_masks.unsqueeze(1))
            if not (len(ip_adapter_masks) == len(self.processor.scale) == len(ip_hidden_states)):
                raise ValueError(
                    f"Length of ip_adapter_masks array ({len(ip_adapter_masks)}) must match "
                    f"length of self.processor.scale array ({len(self.processor.scale)}) and number of ip_hidden_states "
                    f"({len(ip_hidden_states)})"
                )
            else:
                for index, (mask, scale, ip_state) in enumerate(zip(ip_adapter_masks, self.processor.scale, ip_hidden_states)):
                    if mask is None:
                        continue
                    if not isinstance(mask, torch.Tensor) or mask.ndim != 4:
                        raise ValueError(
                            "Each element of the ip_adapter_masks array should be a tensor with shape "
                            "[1, num_images_for_ip_adapter, height, width]."
                            " Please use `IPAdapterMaskProcessor` to preprocess your mask"
                        )
                    if mask.shape[1] != ip_state.shape[1]:
                        raise ValueError(
                            f"Number of masks ({mask.shape[1]}) does not match "
                            f"number of ip images ({ip_state.shape[1]}) at index {index}"
                        )
                    if isinstance(scale, list) and not len(scale) == mask.shape[1]:
                        raise ValueError(
                            f"Number of masks ({mask.shape[1]}) does not match "
                            f"number of scales ({len(scale)}) at index {index}"
                        )
        else:
            ip_adapter_masks = [None] * len(self.processor.scale)

        # for ip-adapter
        for current_ip_hidden_states, scale, to_k_ip, to_v_ip, mask in zip(
            ip_hidden_states, self.processor.scale, self.processor.to_k_ip, self.processor.to_v_ip, ip_adapter_masks
        ):
            skip = False
            if isinstance(scale, list):
                if all(s == 0 for s in scale):
                    skip = True
            elif scale == 0:
                skip = True
            if not skip:
                if mask is not None:
                    if not isinstance(scale, list):
                        scale = [scale] * mask.shape[1]

                    current_num_images = mask.shape[1]
                    for i in range(current_num_images):
                        ip_key = to_k_ip(current_ip_hidden_states[:, i, :, :])
                        ip_value = to_v_ip(current_ip_hidden_states[:, i, :, :])

                        ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                        ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                        # the output of sdp = (batch, num_heads, seq_len, head_dim)
                        # TODO: add support for attn.scale when we move to Torch 2.1
                        _current_ip_hidden_states = F.scaled_dot_product_attention(
                            query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
                        )

                        _current_ip_hidden_states = _current_ip_hidden_states.transpose(1, 2).reshape(
                            batch_size, -1, attn.heads * head_dim
                        )
                        _current_ip_hidden_states = _current_ip_hidden_states.to(query.dtype)

                        mask_downsample = IPAdapterMaskProcessor.downsample(
                            mask[:, i, :, :],
                            batch_size,
                            _current_ip_hidden_states.shape[1],
                            _current_ip_hidden_states.shape[2],
                        )

                        mask_downsample = mask_downsample.to(dtype=query.dtype, device=query.device)
                        hidden_states = hidden_states + scale[i] * (_current_ip_hidden_states * mask_downsample)
                else:
                    current_num_images=current_ip_hidden_states.size()[1]
                    for i in range(current_num_images):
                        ip_key = to_k_ip(current_ip_hidden_states[:, i, :, :])
                        ip_value = to_v_ip(current_ip_hidden_states[:, i, :, :])

                        ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                        ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                        # the output of sdp = (batch, num_heads, seq_len, head_dim)
                        _current_ip_hidden_states = F.scaled_dot_product_attention(
                            query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
                        )

                        _current_ip_hidden_states = _current_ip_hidden_states.transpose(1, 2).reshape(
                            batch_size, -1, attn.heads * head_dim
                        )

                        attn_weight = query @ ip_key.transpose(-2, -1)
                        attn_weight = torch.softmax(attn_weight, dim=-1)

                        self.kv_ip.append(attn_weight)
                        print("added to kv ip" ,len(self.kv_ip),height,width,attn_weight.size())
                        _current_ip_hidden_states = _current_ip_hidden_states.to(query.dtype)

                        
                        hidden_states = hidden_states + scale * (_current_ip_hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    


def get_modules_of_types(model, target_classes):
        return [(name, module) for name, module in model.named_modules()
                if isinstance(module, target_classes)]

def insert_monkey(pipe:StableDiffusionPipeline):
    attn_list=get_modules_of_types(pipe.unet,Attention)
    for name,module in attn_list:
        if getattr(module,"processor",None)!=None and type(getattr(module,"processor",None))==IPAdapterAttnProcessor2_0:
            monkey_processor=MonkeyIPAttnProcessor(module.processor,name)
            setattr(module,"processor",monkey_processor)
            processor_name=name+".processor"
            pipe.unet.attn_processors[processor_name]=monkey_processor

def set_ip_adapter_scale_monkey(self:IPAdapterMixin,scale):
    """
    Set IP-Adapter scales per-transformer block. Input `scale` could be a single config or a list of configs for
    granular control over each IP-Adapter behavior. A config can be a float or a dictionary.

    Example:

    ```py
    # To use original IP-Adapter
    scale = 1.0
    pipeline.set_ip_adapter_scale(scale)

    # To use style block only
    scale = {
        "up": {"block_0": [0.0, 1.0, 0.0]},
    }
    pipeline.set_ip_adapter_scale(scale)

    # To use style+layout blocks
    scale = {
        "down": {"block_2": [0.0, 1.0]},
        "up": {"block_0": [0.0, 1.0, 0.0]},
    }
    pipeline.set_ip_adapter_scale(scale)

    # To use style and layout from 2 reference images
    scales = [{"down": {"block_2": [0.0, 1.0]}}, {"up": {"block_0": [0.0, 1.0, 0.0]}}]
    pipeline.set_ip_adapter_scale(scales)
    ```
    """
    unet = getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet
    if not isinstance(scale, list):
        scale = [scale]
    scale_configs = _maybe_expand_lora_scales(unet, scale, default_scale=0.0)

    for attn_name, attn_processor in unet.attn_processors.items():
        if isinstance(
            attn_processor, (IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0, IPAdapterXFormersAttnProcessor,MonkeyIPAttnProcessor)
        ):
            if len(scale_configs) != len(attn_processor.scale):
                raise ValueError(
                    f"Cannot assign {len(scale_configs)} scale_configs to {len(attn_processor.scale)} IP-Adapter."
                )
            elif len(scale_configs) == 1:
                scale_configs = scale_configs * len(attn_processor.scale)
            for i, scale_config in enumerate(scale_configs):
                if isinstance(scale_config, dict):
                    for k, s in scale_config.items():
                        if attn_name.startswith(k):
                            attn_processor.scale[i] = s
                else:
                    attn_processor.scale[i] = scale_config


def reset_monkey(pipe):
    attn_list=get_modules_of_types(pipe.unet,MonkeyIPAttnProcessor)
    for name,module in attn_list:
        module.reset()
        
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

def get_mask_rect(layer_index:int, 
             attn_list:list,step:int,
             token:int,
             threshold:float,
             latent_dim_x:int,
             latent_dim_y:int,
             kv_type:str="ip",):
    #print("layer",layer_index)
    module=attn_list[layer_index][1] #get the module no name
    #module.processor.kv_ip
    if kv_type=="ip":
        processor_kv=module.processor.kv_ip
    elif kv_type=="str":
        processor_kv=module.processor.kv
    size=processor_kv[step].size()
    print('\tprocessor_kv[step].size()',processor_kv[step].size())
    
    avg=processor_kv[step].mean(dim=1).squeeze(0)
    print("\t avg ", avg.size())

    print("\tlatent",latent_dim_x,latent_dim_y)
    avg=avg.view([latent_dim_x,latent_dim_y,-1])
    print("\t avg ", avg.size())
    avg=avg[:,:,token]
    print("\t avg ", avg.size())
    avg_min,avg_max=avg.min(),avg.max()
    x_norm = (avg - avg_min) / (avg_max - avg_min)  # [0,1]
    x_norm[x_norm < threshold]=0.
    avg = (x_norm * 255)
    #avg=F.interpolate(avg.unsqueeze(0).unsqueeze(0), size=(dim, dim), mode="nearest").squeeze(0).squeeze(0)

    return avg