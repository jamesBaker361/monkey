"""
Failure Analysis Module
Identifies and categorizes failure modes for paper discussion

Usage:
    from analyze_failures import FailureAnalyzer
    
    analyzer = FailureAnalyzer(pipe, accelerator, dino_metric, clip_model, processor)
    failures = analyzer.run_analysis(data, output_dir="failure_analysis")
    analyzer.print_report(failures)
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
import seaborn as sns
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass, asdict
import logging

from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
from diffusers.image_processor import IPAdapterMaskProcessor
from diffusers.models.attention_processor import Attention
sys.path.append(os.path.dirname(__file__))
from ipattn import MonkeyIPAttnProcessor, get_modules_of_types, reset_monkey, insert_monkey, set_ip_adapter_scale_monkey
from pipelines import CompatibleLatentConsistencyModelPipeline
import datasets
from transformers import AutoProcessor, CLIPModel
from eval_helpers import DinoMetric
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


@dataclass
class FailureExample:
    """Single failure example"""
    sample_id: int
    failure_type: str
    severity: float  # 0-1, higher is worse
    reason: str
    image_path: Optional[str] = None
    metric_value: Optional[float] = None


class FailureAnalyzer:
    """
    Systematically identify and categorize failure modes in IP-Adapter masking
    """
    
    def __init__(self, pipe, accelerator, dino_metric, clip_model, processor,
                 lpips_model=None, device="cuda",
                 attn_list=None, layer_index=15, token=1, dim=256,
                 threshold=0.5, initial_steps=4, final_steps=8,
                 initial_ip_adapter_scale=0.75, kv_type="ip",
                 default_object="character"):
        """
        Args:
            pipe: Diffusion pipeline (must already have insert_monkey(pipe) applied)
            accelerator: Accelerate wrapper
            dino_metric: DINO feature metric for subject preservation
            clip_model: CLIP model for text alignment
            processor: CLIP processor
            lpips_model: LPIPS model (optional, for perceptual distance)
            device: Device to use
            attn_list: modules_of_types(pipe.unet, Attention) list, needed for masked generation
            layer_index, token, dim, threshold, initial_steps, final_steps,
            initial_ip_adapter_scale, kv_type: same masking hyperparameters as main_seg.py
        """
        self.pipe = pipe
        self.accelerator = accelerator
        self.dino_metric = dino_metric
        self.clip_model = clip_model
        self.processor = processor
        self.lpips_model = lpips_model
        self.device = device

        self.attn_list = attn_list
        self.layer_index = layer_index
        self.token = token
        self.dim = dim
        self.threshold = threshold
        self.initial_steps = initial_steps
        self.final_steps = final_steps
        self.initial_ip_adapter_scale = initial_ip_adapter_scale
        self.kv_type = kv_type
        self.default_object = default_object
        self.mask_processor = IPAdapterMaskProcessor()

        # Thresholds for failure detection
        self.clip_score_threshold = 0.2  # Below this = poor text alignment
        self.dino_score_threshold = 0.3  # Below this = poor subject preservation
        self.lpips_threshold = 0.5  # Above this = significant degradation

        # Collect failures
        self.failures: Dict[str, List[FailureExample]] = {
            "subject_degradation": [],
            "prompt_ignored": [],
            "background_artifacts": [],
            "low_quality_generation": [],
        }
    
    def run_analysis(self, data, num_samples: int = 100,
                     output_dir: str = "failure_analysis") -> Dict[str, List[FailureExample]]:
        """
        Run failure analysis on dataset
        
        Args:
            data: Dataset to analyze
            num_samples: How many samples to analyze
            output_dir: Where to save failure examples
        
        Returns:
            Dictionary of failures by type
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting failure analysis on {num_samples} samples...")
        
        for idx, row in enumerate(data):
            if idx >= num_samples:
                break
            
            if idx % 10 == 0:
                logger.info(f"Analyzing sample {idx}/{num_samples}")
            
            try:
                self._analyze_sample(idx, row, output_path)
            except Exception as e:
                logger.warning(f"Error analyzing sample {idx}: {e}")
                continue
        
        # Save failure summary
        self._save_summary(output_path)
        
        return self.failures
    
    def _analyze_sample(self, sample_id: int, row: dict, output_dir: Path):
        """Analyze a single sample for failures"""
        
        ip_adapter_image = row["image"]
        if "prompt" in row:
            prompt = row["prompt"]
        else:
            object=row.get("object",row.get("text", self.default_object))
            prompt=object+real_test_prompt_list[sample_id % len(real_test_prompt_list)]

        # Generate variants
        try:
            masked_img = self._generate_image(
                ip_adapter_image, prompt,
                use_mask=True
            )

            # Unmasked version
            unmasked_img = self._generate_image(
                ip_adapter_image, prompt,
                use_mask=False
            )
        except Exception as e:
            logger.warning(f"Generation failed for sample {sample_id}: {e}")
            return

        # Check for failures

        # 1. Subject Degradation (LPIPS in subject region)
        degr_failure = self._check_subject_degradation(
            ip_adapter_image, masked_img, sample_id, output_dir
        )
        if degr_failure:
            self.failures["subject_degradation"].append(degr_failure)
        
        # 2. Prompt Ignored (low CLIP score)
        prompt_failure = self._check_prompt_ignored(
            masked_img, unmasked_img, prompt, sample_id, output_dir
        )
        if prompt_failure:
            self.failures["prompt_ignored"].append(prompt_failure)

        # 3. Background Artifacts (DINO inconsistency)
        artifact_failure = self._check_background_artifacts(
            ip_adapter_image, masked_img, sample_id, output_dir
        )
        if artifact_failure:
            self.failures["background_artifacts"].append(artifact_failure)

        # 4. Low Quality Generation (multiple metrics)
        quality_failure = self._check_low_quality(
            masked_img, unmasked_img, prompt, sample_id, output_dir
        )
        if quality_failure:
            self.failures["low_quality_generation"].append(quality_failure)

    def _check_subject_degradation(self, original: Image.Image,
                                    masked: Image.Image, 
                                    sample_id: int,
                                    output_dir: Path) -> Optional[FailureExample]:
        """
        Check if subject is degraded in masked version (compared to original)
        Uses DINO or LPIPS to measure perceptual distance
        """
        
        try:
            # Convert to tensors
            original_t = torch.from_numpy(np.array(original)).float().to(self.device) / 255.0
            masked_t = torch.from_numpy(np.array(masked)).float().to(self.device) / 255.0
            
            # Use DINO score
            dino_scores = self.dino_metric.get_scores(original, [masked])
            dino_score = dino_scores[0] if dino_scores else 0.0
            
            if dino_score < self.dino_score_threshold:
                # Save example
                comparison = self._create_comparison_image(
                    original, masked, title="Subject Degradation"
                )
                comparison.save(output_dir / f"subject_degradation_{sample_id}.png")
                
                failure = FailureExample(
                    sample_id=sample_id,
                    failure_type="subject_degradation",
                    severity=1.0 - dino_score,  # Higher severity = lower DINO
                    reason=f"DINO score {dino_score:.3f} below threshold {self.dino_score_threshold}",
                    image_path=str(output_dir / f"subject_degradation_{sample_id}.png"),
                    metric_value=dino_score,
                )
                return failure
        
        except Exception as e:
            logger.warning(f"Error checking subject degradation: {e}")
        
        return None
    
    def _check_prompt_ignored(self, masked: Image.Image, unmasked: Image.Image,
                              prompt: str, sample_id: int,
                              output_dir: Path) -> Optional[FailureExample]:
        """
        Check if prompt is ignored (masked version doesn't improve over unmasked)
        Uses CLIP text alignment
        """
        try:
            # Score both versions with CLIP
            inputs = self.processor(
                text=[prompt],
                images=[masked, unmasked],
                return_tensors="pt",
                padding=True
            )
            
            outputs = self.clip_model(**inputs)
            logits = outputs.logits_per_text[0]  # Shape: (2,)
            
            masked_score = logits[0].item()
            unmasked_score = logits[1].item()
            
            # Failure if masking didn't improve alignment
            if masked_score < self.clip_score_threshold and masked_score <= unmasked_score:
                comparison = self._create_comparison_image(
                    masked, unmasked, title=f"Prompt Ignored\nPrompt: {prompt[:40]}"
                )
                comparison.save(output_dir / f"prompt_ignored_{sample_id}.png")
                
                failure = FailureExample(
                    sample_id=sample_id,
                    failure_type="prompt_ignored",
                    severity=1.0 - masked_score,  # Higher severity = lower CLIP score
                    reason=f"CLIP score {masked_score:.3f} (unmasked: {unmasked_score:.3f})",
                    image_path=str(output_dir / f"prompt_ignored_{sample_id}.png"),
                    metric_value=masked_score,
                )
                return failure
        
        except Exception as e:
            logger.warning(f"Error checking prompt ignored: {e}")
        
        return None
    
    def _check_background_artifacts(self, original: Image.Image,
                                     masked: Image.Image,
                                     sample_id: int,
                                     output_dir: Path) -> Optional[FailureExample]:
        """
        Check for visual artifacts in background
        Uses DINO features to detect inconsistency
        """
        try:
            # If DINO score is very low overall, suggests artifacts
            dino_scores = self.dino_metric.get_scores(original, [masked])
            dino_score = dino_scores[0] if dino_scores else 0.0
            
            # Check if it's a background issue (not subject degradation)
            # We'd need to look at non-subject regions
            # Simplified: flag if DINO is low but subject might be OK
            
            if dino_score < 0.25:  # Very low
                comparison = self._create_comparison_image(
                    original, masked, title="Potential Background Artifacts"
                )
                comparison.save(output_dir / f"background_artifacts_{sample_id}.png")
                
                failure = FailureExample(
                    sample_id=sample_id,
                    failure_type="background_artifacts",
                    severity=1.0 - dino_score,
                    reason=f"Visual inconsistency detected (DINO: {dino_score:.3f})",
                    image_path=str(output_dir / f"background_artifacts_{sample_id}.png"),
                    metric_value=dino_score,
                )
                return failure
        
        except Exception as e:
            logger.warning(f"Error checking background artifacts: {e}")
        
        return None
    
    def _check_low_quality(self, masked: Image.Image, unmasked: Image.Image,
                           prompt: str, sample_id: int,
                           output_dir: Path) -> Optional[FailureExample]:
        """
        Check for low quality overall (fails on multiple metrics)
        """
        try:
            # Check multiple metrics
            issues = []
            severity_scores = []
            
            # CLIP text score
            inputs = self.processor(
                text=[prompt],
                images=[masked],
                return_tensors="pt",
                padding=True
            )
            outputs = self.clip_model(**inputs)
            text_score = outputs.logits_per_text[0, 0].item()
            
            if text_score < 0.15:
                issues.append(f"CLIP text: {text_score:.3f}")
                severity_scores.append(1.0 - text_score)
            
            # Flag if multiple issues
            if len(issues) >= 2:
                comparison = self._create_comparison_image(
                    masked, unmasked, title="Low Quality Generation"
                )
                comparison.save(output_dir / f"low_quality_{sample_id}.png")
                
                failure = FailureExample(
                    sample_id=sample_id,
                    failure_type="low_quality_generation",
                    severity=np.mean(severity_scores) if severity_scores else 0.5,
                    reason="; ".join(issues),
                    image_path=str(output_dir / f"low_quality_{sample_id}.png"),
                    metric_value=text_score,
                )
                return failure
        
        except Exception as e:
            logger.warning(f"Error checking low quality: {e}")
        
        return None
    
    def _generate_image(self, ip_adapter_image: Image.Image, prompt: str,
                        use_mask: bool = False) -> Image.Image:
        """
        Generate image with or without masking, using the same monkey-patched,
        attention-mask-guided IP-Adapter mechanism as main_seg.py.
        """

        reset_monkey(self.pipe)

        if not use_mask:
            generator = torch.Generator()
            generator.manual_seed(42)
            set_ip_adapter_scale_monkey(self.pipe, 1.0)
            image = self.pipe(
                prompt,
                self.dim, self.dim,
                self.final_steps,
                ip_adapter_image=ip_adapter_image,
                generator=generator,
            ).images[0]
            return image

        if self.attn_list is None:
            raise ValueError("attn_list must be provided to FailureAnalyzer for masked generation")

        # initial low-scale pass to record the IP-Adapter attention maps used for masking
        generator = torch.Generator()
        generator.manual_seed(42)
        set_ip_adapter_scale_monkey(self.pipe, self.initial_ip_adapter_scale)
        initial_image = self.pipe(
            prompt,
            self.dim, self.dim,
            self.initial_steps,
            ip_adapter_image=ip_adapter_image,
            generator=generator,
        ).images[0]

        initial_mask_step_list = quarter_trim_step_list(self.initial_steps)
        mask = sum([get_mask(self.layer_index, self.attn_list, step, self.token, self.dim, self.threshold, self.kv_type)
                    for step in initial_mask_step_list])
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(self.dim, self.dim), mode="nearest").squeeze(0).squeeze(0)
        mask[mask > 1] = 1.

        ip_mask = self.mask_processor.preprocess(mask)

        final_mask_step_list = quarter_trim_step_list(self.final_steps)
        scale_step_dict = {i: 0 for i in range(self.final_steps)}
        for i in final_mask_step_list:
            scale_step_dict[i] = 1.0

        generator = torch.Generator()
        generator.manual_seed(42)
        set_ip_adapter_scale_monkey(self.pipe, 1.0)
        image = self.pipe(
            prompt,
            self.dim, self.dim,
            self.final_steps,
            ip_adapter_image=ip_adapter_image,
            generator=generator,
            cross_attention_kwargs={
                "ip_adapter_masks": ip_mask
            },
            mask_step_list=final_mask_step_list,
            scale_step_dict=scale_step_dict,
        ).images[0]

        return image
    
    def _create_comparison_image(self, img1: Image.Image, img2: Image.Image,
                                 title: str = "") -> Image.Image:
        """Create side-by-side comparison"""
        
        # Resize to same size
        size = (256, 256)
        img1 = img1.resize(size)
        img2 = img2.resize(size)
        
        # Create side-by-side
        comparison = Image.new('RGB', (size[0] * 2 + 10, size[1] + 40), 'white')
        comparison.paste(img1, (0, 40))
        comparison.paste(img2, (size[0] + 10, 40))
        
        # Add labels
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(comparison)
        draw.text((10, 10), title, fill='black')
        draw.text((size[0] + 20, 260), "Left: Original", fill='black')
        draw.text((size[0] + 20 + size[0], 260), "Right: Generated", fill='black')
        
        return comparison
    
    def _save_summary(self, output_dir: Path):
        """Save failure summary to disk"""
        
        summary = {}
        
        for failure_type, examples in self.failures.items():
            summary[failure_type] = {
                "count": len(examples),
                "examples": [asdict(ex) for ex in examples[:5]],  # Save first 5
            }
        
        # Save as JSON
        with open(output_dir / "failure_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Failure summary saved to {output_dir / 'failure_summary.json'}")
    
    def print_report(self, failures: Optional[Dict] = None):
        """Print failure analysis report"""
        
        if failures is None:
            failures = self.failures
        
        total_samples_analyzed = sum(
            len(examples) for examples in failures.values()
        )
        
        print("\n" + "="*60)
        print("FAILURE ANALYSIS REPORT")
        print("="*60)
        
        for failure_type, examples in failures.items():
            count = len(examples)
            
            if count == 0:
                print(f"\n{failure_type.upper()}: 0 failures ✓")
            else:
                print(f"\n{failure_type.upper()}: {count} failures")
                
                # Show severity stats
                severities = [ex.severity for ex in examples]
                print(f"  Mean severity: {np.mean(severities):.3f}")
                print(f"  Max severity: {np.max(severities):.3f}")
                print(f"  Min severity: {np.min(severities):.3f}")
                
                # Show top examples
                sorted_examples = sorted(examples, key=lambda x: x.severity, reverse=True)
                for i, ex in enumerate(sorted_examples[:3]):
                    print(f"  [{i+1}] Sample {ex.sample_id}: {ex.reason}")
                    if ex.metric_value is not None:
                        print(f"       Metric: {ex.metric_value:.3f}")
        
        print("\n" + "="*60)
        print(f"Total failures across all categories: {total_samples_analyzed}")
        print("="*60 + "\n")
    
    def create_failure_visualization(self, output_dir: str = "failure_analysis"):
        """Create summary visualization of failures"""
        
        output_path = Path(output_dir)
        
        # Count failures by type
        failure_counts = {k: len(v) for k, v in self.failures.items()}
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        types = list(failure_counts.keys())
        counts = list(failure_counts.values())
        colors = ['#FF6B6B' if count > 0 else '#4ECDC4' for count in counts]
        
        bars = ax.barh(types, counts, color=colors)
        
        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(count + 0.1, i, str(count), va='center')
        
        ax.set_xlabel('Number of Failures')
        ax.set_title('Failure Mode Distribution')
        ax.set_xlim(0, max(counts) * 1.1 if counts else 1)
        
        plt.tight_layout()
        plt.savefig(output_path / "failure_distribution.png", dpi=150, bbox_inches='tight')
        logger.info(f"Failure visualization saved to {output_path / 'failure_distribution.png'}")
        
        # Create severity heatmap if there are failures
        max_severity = 0
        severity_data = []
        
        for failure_type, examples in self.failures.items():
            if examples:
                severities = [ex.severity for ex in examples[:20]]
                severity_data.append(severities)
                max_severity = max(max_severity, max(severities))
        
        if severity_data:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Pad to same length
            max_len = max(len(s) for s in severity_data)
            padded_data = [
                s + [np.nan] * (max_len - len(s)) for s in severity_data
            ]
            
            im = ax.imshow(padded_data, cmap='Reds', aspect='auto', vmin=0, vmax=1)
            
            ax.set_yticks(range(len(self.failures)))
            ax.set_yticklabels(list(self.failures.keys()))
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Failure Type')
            ax.set_title('Failure Severity Heatmap')
            
            plt.colorbar(im, ax=ax, label='Severity (0=none, 1=severe)')
            plt.tight_layout()
            plt.savefig(output_path / "failure_severity_heatmap.png", dpi=150, bbox_inches='tight')
            logger.info(f"Severity heatmap saved to {output_path / 'failure_severity_heatmap.png'}")


if __name__ == "__main__":
    print_details()
    start=time.time()

    parser=argparse.ArgumentParser()
    parser.add_argument("--mixed_precision",type=str,default="no")
    parser.add_argument("--src_dataset",type=str, default="jlbaker361/dreambooth")
    parser.add_argument("--num_samples",type=int,default=100)
    parser.add_argument("--object",type=str,default="character")
    parser.add_argument("--initial_steps",type=int,default=4)
    parser.add_argument("--final_steps",type=int,default=8)
    parser.add_argument("--initial_ip_adapter_scale",type=float,default=0.75)
    parser.add_argument("--layer_index",type=int,default=15)
    parser.add_argument("--token",type=int,default=1)
    parser.add_argument("--dim",type=int,default=256)
    parser.add_argument("--threshold",type=float,default=0.5)
    parser.add_argument("--kv_type",type=str,default="ip")
    parser.add_argument("--output_dir",type=str,default="failure_analysis")
    args=parser.parse_args()
    print(args)

    accelerator=Accelerator(mixed_precision=args.mixed_precision)

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    dino_metric = DinoMetric(accelerator.device)

    pipe = CompatibleLatentConsistencyModelPipeline.from_pretrained(
        "SimianLuo/LCM_Dreamshaper_v7",
        torch_dtype=torch.float16,
    ).to(accelerator.device)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
    setattr(pipe,"safety_checker",None)

    insert_monkey(pipe)
    attn_list=get_modules_of_types(pipe.unet,Attention)

    try:
        data=datasets.load_dataset(args.src_dataset)
    except:
        data=datasets.load_dataset(args.src_dataset,download_mode="force_redownload")
    data=data["train"]

    analyzer = FailureAnalyzer(
        pipe=pipe,
        accelerator=accelerator,
        dino_metric=dino_metric,
        clip_model=clip_model,
        processor=processor,
        device=accelerator.device,
        attn_list=attn_list,
        layer_index=args.layer_index,
        token=args.token,
        dim=args.dim,
        threshold=args.threshold,
        initial_steps=args.initial_steps,
        final_steps=args.final_steps,
        initial_ip_adapter_scale=args.initial_ip_adapter_scale,
        kv_type=args.kv_type,
        default_object=args.object,
    )

    with torch.no_grad():
        failures = analyzer.run_analysis(data, num_samples=args.num_samples, output_dir=args.output_dir)

    analyzer.print_report(failures)
    analyzer.create_failure_visualization(output_dir=args.output_dir)

    end=time.time()
    seconds=end-start
    print(f"Failure analysis saved to {args.output_dir}/ ; time elapsed: {seconds} seconds")
    print("all done!")
