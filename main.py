from prism.config import PrismConfig
from prism.metrics import (
    compute_depth_metrics,
    compute_intensity_metrics,
    print_depth_metrics,
    print_intensity_metrics,
)
from prism.pipeline import run_prism_demo
from prism.visualization import show_reconstruction


def main() -> None:
    config = PrismConfig()
    result = run_prism_demo(config)

    if config.plot_results:
        show_reconstruction(
            gt_depth=result.data.gt_depth,
            raw_depth=result.data.raw_depth,
            prism_depth=result.prism_depth,
            gt_intensity=result.data.gt_intensity,
            raw_intensity=result.data.raw_intensity,
            prism_intensity=result.prism_intensity,
        )

    depth_metrics = compute_depth_metrics(
        gt_depth=result.data.gt_depth,
        raw_depth=result.data.raw_depth,
        prism_depth=result.prism_depth,
        norm_bins=config.depth_norm_bins,
    )
    intensity_metrics = compute_intensity_metrics(
        gt_intensity=result.data.gt_intensity,
        raw_intensity=result.data.raw_intensity,
        prism_intensity=result.prism_intensity,
    )

    print_depth_metrics(depth_metrics)
    print_intensity_metrics(intensity_metrics)


if __name__ == "__main__":
    main()
