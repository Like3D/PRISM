from dataclasses import dataclass

import numpy as np
from skimage.metrics import structural_similarity as ssim


@dataclass(frozen=True)
class ReconstructionMetrics:
    rmse: float
    absrel: float | None
    psnr: float
    ssim: float


@dataclass(frozen=True)
class MetricComparison:
    raw: ReconstructionMetrics
    prism: ReconstructionMetrics


def compute_depth_metrics(
    gt_depth: np.ndarray,
    raw_depth: np.ndarray,
    prism_depth: np.ndarray,
    norm_bins: int,
) -> MetricComparison:
    gt_norm = np.asarray(gt_depth) / norm_bins
    raw_norm = np.asarray(raw_depth) / norm_bins
    prism_norm = np.asarray(prism_depth) / norm_bins
    mask = gt_norm > 0

    raw_metrics = _compute_depth_single(gt_norm, raw_norm, mask)
    prism_metrics = _compute_depth_single(gt_norm, prism_norm, mask)
    return MetricComparison(raw=raw_metrics, prism=prism_metrics)


def compute_intensity_metrics(
    gt_intensity: np.ndarray,
    raw_intensity: np.ndarray,
    prism_intensity: np.ndarray,
) -> MetricComparison:
    gt_max = np.max(gt_intensity)
    gt_norm = gt_intensity / gt_max
    raw_norm = raw_intensity / gt_max
    prism_norm = prism_intensity / gt_max

    raw_metrics = _compute_image_single(gt_norm, raw_norm)
    prism_metrics = _compute_image_single(gt_norm, prism_norm)
    return MetricComparison(raw=raw_metrics, prism=prism_metrics)


def print_depth_metrics(metrics: MetricComparison) -> None:
    print("-" * 40)
    print("Depth metrics           Raw      |    PRISM")
    print("-" * 40)
    print(f"RMSE   lower better:  {metrics.raw.rmse:.4f}  |  {metrics.prism.rmse:.4f}")
    print(f"AbsRel lower better:  {metrics.raw.absrel:.4f}  |  {metrics.prism.absrel:.4f}")
    print(f"PSNR   higher better: {metrics.raw.psnr:.4f}  |  {metrics.prism.psnr:.4f}")
    print(f"SSIM   higher better: {metrics.raw.ssim:.4f}  |  {metrics.prism.ssim:.4f}")
    print("-" * 40)


def print_intensity_metrics(metrics: MetricComparison) -> None:
    print("-" * 40)
    print("Intensity metrics       Raw      |    PRISM")
    print("-" * 40)
    print(f"RMSE   lower better:  {metrics.raw.rmse:.4f}  |  {metrics.prism.rmse:.4f}")
    print(f"PSNR   higher better: {metrics.raw.psnr:.4f}  |  {metrics.prism.psnr:.4f}")
    print(f"SSIM   higher better: {metrics.raw.ssim:.4f}  |  {metrics.prism.ssim:.4f}")
    print("-" * 40)


def _compute_depth_single(
    gt_norm: np.ndarray,
    estimate_norm: np.ndarray,
    mask: np.ndarray,
) -> ReconstructionMetrics:
    mse = np.mean((estimate_norm[mask] - gt_norm[mask]) ** 2)
    rmse = float(np.sqrt(mse))
    absrel = float(np.mean(np.abs(estimate_norm[mask] - gt_norm[mask]) / gt_norm[mask]))
    data_max = np.max(gt_norm[mask])
    psnr = float(10 * np.log10((data_max**2) / mse)) if mse != 0 else float("inf")

    gt_masked = gt_norm * mask
    estimate_masked = estimate_norm * mask
    data_range = np.max(gt_masked) - np.min(gt_masked)
    ssim_value = float(ssim(gt_masked, estimate_masked, win_size=7, data_range=data_range))
    return ReconstructionMetrics(rmse=rmse, absrel=absrel, psnr=psnr, ssim=ssim_value)


def _compute_image_single(gt_norm: np.ndarray, estimate_norm: np.ndarray) -> ReconstructionMetrics:
    mse = np.mean((estimate_norm - gt_norm) ** 2)
    rmse = float(np.sqrt(mse))
    data_max = np.max(gt_norm)
    psnr = float(10 * np.log10((data_max**2) / mse)) if mse != 0 else float("inf")
    data_range = np.max(gt_norm) - np.min(gt_norm)
    ssim_value = float(ssim(gt_norm, estimate_norm, win_size=7, data_range=data_range))
    return ReconstructionMetrics(rmse=rmse, absrel=None, psnr=psnr, ssim=ssim_value)
