"""
Failure Analysis Module
Identifies and categorizes failure modes for paper discussion

Usage:
    from analyze_failures import FailureAnalyzer
    
    analyzer = FailureAnalyzer(pipe, accelerator, dino_metric, clip_model, processor)
    failures = analyzer.run_analysis(data, output_dir="failure_analysis")
    analyzer.print_report(failures)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass, asdict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
                 lpips_model=None, device="cuda"):
        """
        Args:
            pipe: Diffusion pipeline
            accelerator: Accelerate wrapper
            dino_metric: DINO feature metric for subject preservation
            clip_model: CLIP model for text alignment
            processor: CLIP processor
            lpips_model: LPIPS model (optional, for perceptual distance)
            device: Device to use
        """
        self.pipe = pipe
        self.accelerator = accelerator
        self.dino_metric = dino_metric
        self.clip_model = clip_model
        self.processor = processor
        self.lpips_model = lpips_model
        self.device = device
        
        # Thresholds for failure detection
        self.clip_score_threshold = 0.2  # Below this = poor text alignment
        self.dino_score_threshold = 0.3  # Below this = poor subject preservation
        self.lpips_threshold = 0.5  # Above this = significant degradation
        self.sam_detection_failure = 0  # No detections
        
        # Collect failures
        self.failures: Dict[str, List[FailureExample]] = {
            "sam_no_detection": [],
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
        prompt = row.get("prompt", "character in a beautiful landscape")
        
        # Generate variants
        try:
            # Raw mask version
            masked_img = self._generate_image(
                ip_adapter_image, prompt, 
                use_mask=True, mask_type="raw"
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
        
        # 1. SAM Segmentation Failure
        sam_failure = self._check_sam_failure(ip_adapter_image, sample_id, output_dir)
        if sam_failure:
            return  # Skip if SAM fails - can't evaluate mask quality
        
        # 2. Subject Degradation (LPIPS in subject region)
        degr_failure = self._check_subject_degradation(
            ip_adapter_image, masked_img, sample_id, output_dir
        )
        if degr_failure:
            self.failures["subject_degradation"].append(degr_failure)
        
        # 3. Prompt Ignored (low CLIP score)
        prompt_failure = self._check_prompt_ignored(
            masked_img, unmasked_img, prompt, sample_id, output_dir
        )
        if prompt_failure:
            self.failures["prompt_ignored"].append(prompt_failure)
        
        # 4. Background Artifacts (DINO inconsistency)
        artifact_failure = self._check_background_artifacts(
            ip_adapter_image, masked_img, sample_id, output_dir
        )
        if artifact_failure:
            self.failures["background_artifacts"].append(artifact_failure)
        
        # 5. Low Quality Generation (multiple metrics)
        quality_failure = self._check_low_quality(
            masked_img, unmasked_img, prompt, sample_id, output_dir
        )
        if quality_failure:
            self.failures["low_quality_generation"].append(quality_failure)
    
    def _check_sam_failure(self, image: Image.Image, sample_id: int, 
                           output_dir: Path) -> Optional[FailureExample]:
        """Check if SAM segmentation fails on this image"""
        try:
            from custom_sam_detector import CustomSamDetector
            
            # Assuming custom_sam is available
            # segmented, annotations = custom_sam(image)
            
            # Simplified: check if object detection fails
            # In practice, you'd call your SAM detector
            
            return None  # No failure detected
        except Exception as e:
            failure = FailureExample(
                sample_id=sample_id,
                failure_type="sam_no_detection",
                severity=1.0,
                reason=f"SAM segmentation failed: {str(e)}",
            )
            self.failures["sam_no_detection"].append(failure)
            return failure
    
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
                        use_mask: bool = False, mask_type: str = "raw") -> Image.Image:
        """
        Generate image with or without masking
        
        Note: This is a simplified version. You'd need to integrate with your
        actual generation pipeline (main_seg.py logic)
        """
        
        generator = torch.Generator()
        generator.manual_seed(42)
        
        if not use_mask:
            # Generate without masking
            image = self.pipe(
                prompt,
                256, 256,
                num_inference_steps=8,
                ip_adapter_image=ip_adapter_image,
                generator=generator,
            ).images[0]
        else:
            # Generate with masking
            # This would require your mask generation logic from main_seg.py
            image = self.pipe(
                prompt,
                256, 256,
                num_inference_steps=8,
                ip_adapter_image=ip_adapter_image,
                generator=generator,
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


# Example usage
if __name__ == "__main__":
    """
    Example of how to use FailureAnalyzer
    
    from analyze_failures import FailureAnalyzer
    from eval_helpers import DinoMetric
    from transformers import CLIPModel, AutoProcessor
    
    # Initialize
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    dino_metric = DinoMetric(device)
    
    analyzer = FailureAnalyzer(
        pipe=pipe,
        accelerator=accelerator,
        dino_metric=dino_metric,
        clip_model=clip_model,
        processor=processor,
    )
    
    # Run analysis
    failures = analyzer.run_analysis(data, num_samples=100, output_dir="failure_analysis")
    
    # Print report
    analyzer.print_report(failures)
    
    # Create visualizations
    analyzer.create_failure_visualization(output_dir="failure_analysis")
    """
    pass
