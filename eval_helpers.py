from typing import List,Optional,Union

import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms
from transformers import AutoModel
from PIL import Image


class DinoMetric:
    def __init__(self,device:str,model_name:str="facebook/dino-vits16"):
        self.device=device
        self.dino_model=AutoModel.from_pretrained(model_name, add_pooling_layer=False).to(device)
        self.T = transforms.Compose(
            [
                transforms.Resize(256, interpolation=3),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
            )

    def embed_images(self, image_list:Union[Image.Image, List[Image.Image]])-> torch.Tensor:
        if type(image_list)!=list:
            image_list=[image_list]
        image_tensor_list=torch.stack([self.T(image.convert("RGB")).to(self.device) for image in image_list])
        return self.dino_model(image_tensor_list).last_hidden_state[:,0,:]
    
    @torch.no_grad()
    def get_scores(self,src_image,generated_image_list)->list:
        src_embedding=self.embed_images(src_image)
        generated_embedding=self.embed_images(generated_image_list)

        cosine_similarities=F.cosine_similarity(src_embedding,generated_embedding)

        return cosine_similarities.detach().cpu().numpy()
    