from typing import List,Optional,Union,Dict

import torch
import lpips
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms
from transformers import AutoModel
from PIL import Image


class SubjectPreservationMetric:
    """
    Measure how well the subject is preserved vs background separation.
    This is the key metric reviewers will ask for.
    """
    
    def __init__(self, device="cuda"):
        self.device = device
        self.lpips_model = lpips.LPIPS(net='alex', version='0.1').to(device).eval()
        
    def compute_preservation_score(self, 
                                   original_img: torch.Tensor,
                                   generated_img: torch.Tensor,
                                   subject_mask: torch.Tensor) -> Dict[str, float]:
        """
        Args:
            original_img: Source image (3, H, W), range [0, 1]
            generated_img: Generated image (3, H, W), range [0, 1]
            subject_mask: Binary mask of subject region (1, H, W)
        
        Returns:
            Dict with metrics:
            - subject_preservation: LPIPS distance in subject region (lower=better)
            - background_divergence: LPIPS distance in background (higher=better for change)
            - trade_off_ratio: preservation vs divergence balance
        """
        
        # Ensure on device
        original_img = original_img.to(self.device).unsqueeze(0)  # (1, 3, H, W)
        generated_img = generated_img.to(self.device).unsqueeze(0)  # (1, 3, H, W)
        subject_mask = subject_mask.to(self.device).unsqueeze(0)  # (1, 1, H, W)
        bg_mask = 1 - subject_mask
        
        with torch.no_grad():
            # Full LPIPS score
            total_lpips = self.lpips_model(original_img * 2 - 1, 
                                           generated_img * 2 - 1).item()
            
            # Subject region LPIPS (masking)
            if subject_mask.sum() > 100:  # Only if mask has meaningful area
                subject_lpips = self.lpips_model(
                    (original_img * subject_mask) * 2 - 1,
                    (generated_img * subject_mask) * 2 - 1
                ).item()
            else:
                subject_lpips = 0.0
            
            # Background LPIPS
            if bg_mask.sum() > 100:
                bg_lpips = self.lpips_model(
                    (original_img * bg_mask) * 2 - 1,
                    (generated_img * bg_mask) * 2 - 1
                ).item()
            else:
                bg_lpips = 0.0
        
        return {
            "subject_preservation": 1.0 - subject_lpips,  # Higher is better
            "background_divergence": bg_lpips,             # Higher is better (allow change)
            "trade_off_ratio": (1.0 - subject_lpips) / max(bg_lpips, 0.01),
            "total_lpips": total_lpips,
        }



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
    