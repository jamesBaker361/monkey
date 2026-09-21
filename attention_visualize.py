"""
Attention Visualization Module
Visualizes attention maps to explain the masking mechanism

This module creates interpretable visualizations showing:
1. Text/latent attention patterns
2. IP-Adapter attention patterns  
3. Derived masks from attention
4. Comparison between methods

Usage:
    from attention_visualize import AttentionVisualizer
    
    visualizer = AttentionVisualizer(output_dir="attention_viz")
    visualizer.visualize_inference_steps(pipe, image, prompt, num_steps=8)
    visualizer.create_comparison_grid(initial_image, mask, final_image, prompt)
"""

import os
import sys
import time
import math
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
from diffusers.image_processor import IPAdapterMaskProcessor
from diffusers.models.attention_processor import Attention
sys.path.append(os.path.dirname(__file__))
from ipattn import MonkeyIPAttnProcessor, get_modules_of_types, reset_monkey, insert_monkey, set_ip_adapter_scale_monkey
from pipelines import CompatibleLatentConsistencyModelPipeline
from custom_sam_detector import CustomSamDetector
import datasets
from prompt_list import real_test_prompt_list

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_mask(layer_index:int, attn_list:list, step:int, token:int, dim:int,
             threshold:float, kv_type:str="ip"):
    """Derive an IP-Adapter attention mask, same mechanism as main_seg.py"""
    module=attn_list[layer_index][1]
    processor_kv=module.processor.kv_ip if kv_type=="ip" else module.processor.kv
    avg=processor_kv[step].mean(dim=1).squeeze(0)
    latent_dim=int(math.sqrt(avg.size()[0]))
    avg=avg.view([latent_dim,latent_dim,-1])
    avg=avg[:,:,token]
    avg_min,avg_max=avg.min(),avg.max()
    x_norm = (avg - avg_min) / (avg_max - avg_min)
    x_norm[x_norm < threshold]=0.
    return x_norm * 255


def quarter_trim_step_list(step_count:int)->list:
    step_list=[f for f in range(step_count)]
    quarter=step_count//4
    if quarter>0:
        step_list=step_list[quarter:-quarter]
    return step_list


class AttentionVisualizer:
    """
    Visualizes attention patterns from MonkeyIPAttnProcessor
    """
    
    def __init__(self, output_dir: str = "attention_visualization"):
        """
        Args:
            output_dir: Where to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color schemes
        self.cmap_heat = 'hot'
        self.cmap_coolwarm = 'coolwarm'
        self.cmap_viridis = 'viridis'
    
    def visualize_attention_maps(self, attn_processor, step_idx: int,
                                 layer_idx: int,
                                 save_name: str = "attention_map") -> Optional[np.ndarray]:
        """
        Visualize attention weights from MonkeyIPAttnProcessor
        
        Args:
            attn_processor: MonkeyIPAttnProcessor instance (has kv and kv_ip)
            step_idx: Which inference step to visualize
            layer_idx: Which attention layer
            save_name: Name for saved figure
        
        Returns:
            Attention map as numpy array (or None if not available)
        """
        
        # Check if we have attention data
        if step_idx >= len(attn_processor.kv_ip):
            logger.warning(f"No attention data for step {step_idx}")
            return None
        
        # Get attention weights
        # Shape: (batch_size, num_heads, seq_len_query, seq_len_key)
        kv_ip = attn_processor.kv_ip[step_idx]  # IP-Adapter attention
        kv = attn_processor.kv[step_idx]  # Text attention
        
        # Move to CPU and detach
        kv_ip = kv_ip.detach().cpu().numpy()
        kv = kv.detach().cpu().numpy()
        
        # Average over batch and heads
        kv_ip_avg = kv_ip[0].mean(axis=0)  # (seq_len_query, seq_len_key)
        kv_avg = kv[0].mean(axis=0)  # (seq_len_query, seq_len_key)
        
        # Get spatial dimension
        seq_len = kv_ip_avg.shape[0]
        spatial_size = int(np.sqrt(seq_len))
        
        if spatial_size * spatial_size != seq_len:
            logger.warning(f"Sequence length {seq_len} is not a perfect square")
            return None
        
        # Focus on subject token (usually token 0 or 1)
        subject_token_idx = 0
        
        # Reshape to spatial
        attention_to_subject_ip = kv_ip_avg[:, subject_token_idx]
        attention_to_subject_ip = attention_to_subject_ip[:spatial_size*spatial_size]
        attention_to_subject_ip = attention_to_subject_ip.reshape(spatial_size, spatial_size)
        
        attention_to_subject_text = kv_avg[:, subject_token_idx]
        attention_to_subject_text = attention_to_subject_text[:spatial_size*spatial_size]
        attention_to_subject_text = attention_to_subject_text.reshape(spatial_size, spatial_size)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: IP-Adapter attention
        im1 = axes[0].imshow(attention_to_subject_ip, cmap=self.cmap_heat, interpolation='bilinear')
        axes[0].set_title(f'IP-Adapter Attention (Step {step_idx})\nFocus on Subject Token', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Spatial X')
        axes[0].set_ylabel('Spatial Y')
        plt.colorbar(im1, ax=axes[0], label='Attention Weight')
        
        # Plot 2: Text attention
        im2 = axes[1].imshow(attention_to_subject_text, cmap=self.cmap_heat, interpolation='bilinear')
        axes[1].set_title(f'Text Attention (Step {step_idx})\nFocus on Subject Token', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Spatial X')
        axes[1].set_ylabel('Spatial Y')
        plt.colorbar(im2, ax=axes[1], label='Attention Weight')
        
        # Plot 3: Difference (what masking targets)
        diff = attention_to_subject_ip - attention_to_subject_text
        im3 = axes[2].imshow(diff, cmap=self.cmap_coolwarm, interpolation='bilinear')
        axes[2].set_title(f'IP-Adapter vs Text (Step {step_idx})\nMask targets high IP attention', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Spatial X')
        axes[2].set_ylabel('Spatial Y')
        plt.colorbar(im3, ax=axes[2], label='Attention Difference')
        
        plt.tight_layout()
        plt.savefig(
            self.output_dir / f"{save_name}_step_{step_idx:02d}.png",
            dpi=150, bbox_inches='tight'
        )
        plt.close()
        
        logger.info(f"Saved attention visualization: {save_name}_step_{step_idx:02d}.png")
        
        return attention_to_subject_ip
    
    def visualize_all_steps(self, attn_processor, num_steps: int,
                           layer_idx: int = 0, save_prefix: str = "attention"):
        """
        Visualize attention for all inference steps
        
        Args:
            attn_processor: Processor with stored attention
            num_steps: How many steps to visualize
            layer_idx: Which layer (usually use mid-point layer)
            save_prefix: Prefix for saved files
        """
        
        for step in range(min(num_steps, len(attn_processor.kv_ip))):
            self.visualize_attention_maps(attn_processor, step, layer_idx, save_prefix)
        
        # Create animation/grid of all steps
        self._create_steps_grid(num_steps, save_prefix)
    
    def _create_steps_grid(self, num_steps: int, prefix: str = "attention"):
        """Create grid showing all steps side-by-side"""
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for step in range(min(num_steps, 8)):
            ax = axes[step]
            
            # Load image (simplified, would need actual data)
            try:
                img_path = self.output_dir / f"{prefix}_step_{step:02d}.png"
                if img_path.exists():
                    img = Image.open(img_path)
                    ax.imshow(img)
                    ax.set_title(f'Step {step}')
                    ax.axis('off')
            except:
                pass
        
        # Hide unused axes
        for ax in axes[num_steps:]:
            ax.axis('off')
        
        plt.suptitle('Attention Maps Across Inference Steps', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{prefix}_all_steps_grid.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved steps grid: {prefix}_all_steps_grid.png")
    
    def create_mechanism_explanation(self, image: Image.Image, 
                                     mask: torch.Tensor,
                                     initial_generation: Image.Image,
                                     raw_mask_generation: Image.Image,
                                     seg_mask_generation: Image.Image,
                                     unmasked_generation: Image.Image,
                                     prompt: str = "in a beautiful landscape"):
        """
        Create a comprehensive figure explaining the mechanism
        
        Shows the pipeline: Original → Attention → Mask → Final Output
        
        Args:
            image: Source IP-Adapter image
            mask: Derived mask (torch tensor or PIL)
            initial_generation: First pass generation (to extract mask)
            raw_mask_generation: Using raw attention-based mask
            seg_mask_generation: Using SAM-guided mask
            unmasked_generation: Without masking (baseline)
            prompt: Text prompt used
        """
        
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(3, 5, figure=fig, hspace=0.3, wspace=0.3)
        
        # Row 1: Input and initial generation
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image)
        ax1.set_title('Source Image\n(for IP-Adapter)', fontsize=11, fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(initial_generation)
        ax2.set_title('Initial Generation\n(Pass 1: 4 steps)', fontsize=11, fontweight='bold')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 2:4])
        ax3.imshow(initial_generation)
        ax3.text(0.5, -0.15, 'Extract Attention Weights\nfrom IP-Adapter', 
                transform=ax3.transAxes, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        ax3.axis('off')
        ax3.set_title('Mechanism: IP-Adapter Attention', fontsize=11, fontweight='bold')
        
        ax4 = fig.add_subplot(gs[0, 4])
        ax4.text(0.5, 0.5, '→', transform=ax4.transAxes, ha='center', va='center',
                fontsize=40, fontweight='bold')
        ax4.axis('off')
        
        # Row 2: Mask derivation
        ax5 = fig.add_subplot(gs[1, 0])
        ax5.imshow(mask.cpu().numpy() if torch.is_tensor(mask) else mask, cmap='gray')
        ax5.set_title('Derived Mask\n(Attention-based)', fontsize=11, fontweight='bold')
        ax5.axis('off')
        
        # Visualization of mask application
        ax6 = fig.add_subplot(gs[1, 1])
        ax6.text(0.5, 0.7, 'Mask Creation:\n\n1. Get IP-Adapter\n   attention\n2. Normalize\n   (0-1)\n3. Threshold\n   (0.5)',
                transform=ax6.transAxes, ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax6.axis('off')
        ax6.set_title('Mask Derivation', fontsize=11, fontweight='bold')
        
        ax7 = fig.add_subplot(gs[1, 2])
        ax7.text(0.5, 0.5, '→', transform=ax7.transAxes, ha='center', va='center',
                fontsize=40, fontweight='bold')
        ax7.axis('off')
        
        ax8 = fig.add_subplot(gs[1, 3])
        ax8.text(0.5, 0.7, 'Second Pass:\n\n1. Apply mask to\n   IP-Adapter tokens\n2. Restrict to\n   subject region\n3. Let text expand\n   to background',
                transform=ax8.transAxes, ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        ax8.axis('off')
        ax8.set_title('Mask Application', fontsize=11, fontweight='bold')
        
        ax9 = fig.add_subplot(gs[1, 4])
        ax9.text(0.5, 0.5, '→', transform=ax9.transAxes, ha='center', va='center',
                fontsize=40, fontweight='bold')
        ax9.axis('off')
        
        # Row 3: Outputs
        ax10 = fig.add_subplot(gs[2, 0])
        ax10.imshow(unmasked_generation)
        ax10.set_title('Baseline\n(No Mask)', fontsize=11, fontweight='bold')
        ax10.axis('off')
        
        ax11 = fig.add_subplot(gs[2, 1])
        ax11.imshow(raw_mask_generation)
        ax11.set_title('With Raw Mask\n(Attention-based)', fontsize=11, fontweight='bold')
        ax11.axis('off')
        
        ax12 = fig.add_subplot(gs[2, 2])
        ax12.imshow(seg_mask_generation)
        ax12.set_title('With Seg Mask\n(SAM-refined)', fontsize=11, fontweight='bold')
        ax12.axis('off')
        
        # Text explanation
        ax13 = fig.add_subplot(gs[2, 3:])
        explanation = f"""
        Pipeline Explanation:
        
        1. INITIAL PASS: Generate with IP-Adapter (4 steps)
           • Extract IP-Adapter attention weights
           
        2. MASK DERIVATION: Create mask from attention
           • Take IP-Adapter's learned focus region
           • Normalize to 0-1 range
           • Threshold to binary mask
           
        3. MASK REFINEMENT (Optional): Use SAM segmentation
           • Refine mask with semantic segmentation
           • Improves boundary quality
           
        4. SECOND PASS: Generate with masked IP-Adapter
           • Apply mask to restrict IP-Adapter tokens
           • Subject stays focused on original
           • Text prompt free to modify background
           
        Result: Subject preservation + Prompt alignment!
        Prompt: "{prompt[:50]}..."
        """
        
        ax13.text(0.05, 0.95, explanation, transform=ax13.transAxes,
                 fontsize=10, verticalalignment='top', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax13.axis('off')
        
        fig.suptitle('IP-Adapter Masking Mechanism', fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig(self.output_dir / "mechanism_explanation.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info("Saved mechanism explanation figure")
    
    def create_comparison_triplet(self, image: Image.Image,
                                  method1: Image.Image,
                                  method2: Image.Image,
                                  method3: Image.Image,
                                  labels: List[str] = None,
                                  save_name: str = "comparison"):
        """
        Create comparison of three generation methods
        
        Args:
            image: Original image
            method1, method2, method3: Generated images to compare
            labels: Labels for each method
            save_name: Name for saved file
        """
        
        if labels is None:
            labels = ['Method 1', 'Method 2', 'Method 3']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Resize to same size
        size = (256, 256)
        image_r = image.resize(size)
        m1 = method1.resize(size)
        m2 = method2.resize(size)
        m3 = method3.resize(size)
        
        axes[0, 0].imshow(image_r)
        axes[0, 0].set_title('Source Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(m1)
        axes[0, 1].set_title(labels[0], fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(m2)
        axes[1, 0].set_title(labels[1], fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(m3)
        axes[1, 1].set_title(labels[2], fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{save_name}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved comparison: {save_name}.png")
    
    def create_attention_heatmap_overlay(self, image: Image.Image,
                                        attention_map: np.ndarray,
                                        mask: Optional[torch.Tensor] = None,
                                        save_name: str = "attention_overlay") -> Image.Image:
        """
        Overlay attention heatmap on original image
        
        Args:
            image: Original image to overlay on
            attention_map: Attention weights (2D numpy array)
            mask: Optional mask to show boundary
            save_name: Name for saved file
        
        Returns:
            Overlaid image
        """
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Attention heatmap
        axes[1].imshow(image, alpha=0.3)
        im = axes[1].imshow(attention_map, cmap='hot', alpha=0.7, interpolation='bilinear')
        axes[1].set_title('Attention Heatmap Overlay', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        # Mask overlay (if provided)
        if mask is not None:
            mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask
            axes[2].imshow(image, alpha=0.3)
            axes[2].imshow(mask_np, cmap='binary', alpha=0.7)
            axes[2].set_title('Derived Mask', fontsize=12, fontweight='bold')
            axes[2].axis('off')
        else:
            axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{save_name}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved attention overlay: {save_name}.png")
        
        return image
    
    def create_step_progression(self, images: List[Image.Image],
                               captions: List[str],
                               save_name: str = "progression"):
        """
        Create a horizontal strip showing progression through steps
        
        Args:
            images: List of images at each step
            captions: Captions for each image
            save_name: Name for saved file
        """
        
        # Resize all to same size
        size = (200, 200)
        images = [img.resize(size) for img in images]
        
        # Create horizontal concatenation
        total_width = sum(img.width for img in images) + 10 * (len(images) - 1)
        max_height = max(img.height for img in images) + 60
        
        result = Image.new('RGB', (total_width, max_height), color='white')
        
        x_offset = 0
        for img, caption in zip(images, captions):
            result.paste(img, (x_offset, 50))
            
            # Add caption
            draw = ImageDraw.Draw(result)
            draw.text((x_offset + 10, 10), caption, fill='black')
            
            x_offset += img.width + 10
        
        result.save(self.output_dir / f"{save_name}.png")
        logger.info(f"Saved progression: {save_name}.png")
        
        return result


# Example usage and helper function
def visualize_inference_process(pipe, attn_processors: Dict, 
                                image: Image.Image, prompt: str,
                                num_steps: int = 8,
                                output_dir: str = "attention_visualization"):
    """
    Complete visualization of inference process
    
    This would be called from main_seg.py after generation
    
    Example:
        visualize_inference_process(
            pipe, 
            attn_processors={'mid_block': processor},
            image=ip_adapter_image,
            prompt="in a beautiful landscape",
            num_steps=8
        )
    """
    
    visualizer = AttentionVisualizer(output_dir)
    
    logger.info("Visualizing inference process...")
    
    for layer_name, processor in attn_processors.items():
        logger.info(f"Visualizing {layer_name}...")
        visualizer.visualize_all_steps(processor, num_steps, save_prefix=layer_name)
    
    logger.info(f"Visualizations saved to {output_dir}/")


if __name__ == "__main__":
    print_details()
    start=time.time()

    parser=argparse.ArgumentParser()
    parser.add_argument("--mixed_precision",type=str,default="no")
    parser.add_argument("--src_dataset",type=str, default="jlbaker361/ssl-league_captioned_splash-1000-sana")
    parser.add_argument("--sample_index",type=int,default=0,help="which row of src_dataset to use as the sample image")
    parser.add_argument("--object",type=str,default="character")
    parser.add_argument("--sample_prompt",type=str,default=None,help="overrides the prompt built from --object and the dataset row")
    parser.add_argument("--initial_steps",type=int,default=4)
    parser.add_argument("--final_steps",type=int,default=8)
    parser.add_argument("--initial_ip_adapter_scale",type=float,default=0.75)
    parser.add_argument("--layer_index",type=int,default=15)
    parser.add_argument("--token",type=int,default=1)
    parser.add_argument("--dim",type=int,default=256)
    parser.add_argument("--threshold",type=float,default=0.5)
    parser.add_argument("--overlap_frac",type=float,default=0.8)
    parser.add_argument("--kv_type",type=str,default="ip")
    parser.add_argument("--output_dir",type=str,default="attention_visualization")
    args=parser.parse_args()
    print(args)

    accelerator=Accelerator(mixed_precision=args.mixed_precision)

    pipe = CompatibleLatentConsistencyModelPipeline.from_pretrained(
        "SimianLuo/LCM_Dreamshaper_v7",
        torch_dtype=torch.float16,
    ).to(accelerator.device)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
    setattr(pipe,"safety_checker",None)

    insert_monkey(pipe)
    attn_list=get_modules_of_types(pipe.unet,Attention)
    mask_processor=IPAdapterMaskProcessor()
    custom_sam=CustomSamDetector.from_pretrained("ybelkada/segment-anything", subfolder="checkpoints").to(accelerator.device)

    try:
        data=datasets.load_dataset(args.src_dataset)
    except:
        data=datasets.load_dataset(args.src_dataset,download_mode="force_redownload")
    data=data["train"]

    row=data[args.sample_index]
    ip_adapter_image=row["image"]
    object=args.object
    if "object" in row:
        object=row["object"]
    prompt=args.sample_prompt or object+real_test_prompt_list[args.sample_index % len(real_test_prompt_list)]
    print("prompt",prompt)

    with torch.no_grad():
        reset_monkey(pipe)

        # initial low-scale pass to record the IP-Adapter attention maps used for masking
        generator=torch.Generator()
        generator.manual_seed(123)
        set_ip_adapter_scale_monkey(pipe,args.initial_ip_adapter_scale)
        initial_image=pipe(prompt,args.dim,args.dim,args.initial_steps,ip_adapter_image=ip_adapter_image,generator=generator).images[0]

        initial_mask_step_list=quarter_trim_step_list(args.initial_steps)
        mask=sum([get_mask(args.layer_index,attn_list,step,args.token,args.dim,args.threshold,args.kv_type) for step in initial_mask_step_list])
        mask=F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(args.dim, args.dim), mode="nearest").squeeze(0).squeeze(0)
        mask[mask>1]=1.
        ip_mask=mask_processor.preprocess(mask)

        final_mask_step_list=quarter_trim_step_list(args.final_steps)
        scale_step_dict={i:0 for i in range(args.final_steps)}
        for i in final_mask_step_list:
            scale_step_dict[i]=1.0

        # raw attention-mask guided pass
        generator=torch.Generator()
        generator.manual_seed(123)
        set_ip_adapter_scale_monkey(pipe,1.0)
        raw_mask_image=pipe(prompt,args.dim,args.dim,args.final_steps,ip_adapter_image=ip_adapter_image,generator=generator,cross_attention_kwargs={
            "ip_adapter_masks":ip_mask
        }, mask_step_list=final_mask_step_list,scale_step_dict=scale_step_dict).images[0]

        # unmasked baseline pass
        generator=torch.Generator()
        generator.manual_seed(123)
        set_ip_adapter_scale_monkey(pipe,1.0)
        unmasked_image=pipe(prompt,args.dim,args.dim,args.final_steps,ip_adapter_image=ip_adapter_image,generator=generator,
                             scale_step_dict=scale_step_dict).images[0]

        # SAM-refined mask pass
        segmented_image,map_list=custom_sam(initial_image,detect_resolution=args.dim)
        mask_cpu=mask.cpu()
        map_mask=torch.zeros((args.dim,args.dim))
        for ann in map_list:
            map_=torch.from_numpy(ann["segmentation"]).cpu()
            n_ones=map_.sum()
            merged=map_*mask_cpu
            if merged.sum()>=args.overlap_frac*n_ones:
                map_mask=torch.max(map_,map_mask)
        for _ in range(2):
            if len(map_mask.size())>2:
                map_mask=map_mask.squeeze(0)
        ip_map_mask=mask_processor.preprocess(map_mask)

        generator=torch.Generator()
        generator.manual_seed(123)
        set_ip_adapter_scale_monkey(pipe,1.0)
        seg_mask_image=pipe(prompt,args.dim,args.dim,args.final_steps,ip_adapter_image=ip_adapter_image,generator=generator,cross_attention_kwargs={
            "ip_adapter_masks":ip_map_mask
        }, mask_step_list=final_mask_step_list,scale_step_dict=scale_step_dict).images[0]

    visualizer=AttentionVisualizer(output_dir=args.output_dir)

    monkey_processor=attn_list[args.layer_index][1].processor
    step_to_show=initial_mask_step_list[len(initial_mask_step_list)//2] if initial_mask_step_list else 0
    attention_map=visualizer.visualize_attention_maps(monkey_processor, step_idx=step_to_show, layer_idx=args.layer_index)

    resized_source=ip_adapter_image.convert("RGB").resize((args.dim,args.dim))

    if attention_map is not None:
        visualizer.create_attention_heatmap_overlay(resized_source, attention_map, mask=mask_cpu, save_name="attention_overlay")

    visualizer.create_mechanism_explanation(
        image=resized_source,
        mask=mask_cpu,
        initial_generation=initial_image,
        raw_mask_generation=raw_mask_image,
        seg_mask_generation=seg_mask_image,
        unmasked_generation=unmasked_image,
        prompt=prompt,
    )

    visualizer.create_comparison_triplet(
        resized_source, unmasked_image, raw_mask_image, seg_mask_image,
        labels=["Unmasked","Raw Mask","Seg Mask"],
        save_name="comparison_triplet"
    )

    end=time.time()
    seconds=end-start
    print(f"Visualizations saved to {args.output_dir}/ ; time elapsed: {seconds} seconds")
    print("all done!")
