import matplotlib.pyplot as plt
import numpy as np


def show_reconstruction(
    gt_depth: np.ndarray,
    raw_depth: np.ndarray,
    prism_depth: np.ndarray,
    gt_intensity: np.ndarray,
    raw_intensity: np.ndarray,
    prism_intensity: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    depth_min, depth_max = np.min(gt_depth), np.max(gt_depth)

    depth_items = [
        ("Depth Ground Truth", gt_depth, None),
        ("Depth Raw Argmax", raw_depth, None),
        ("Depth PRISM", prism_depth, None),
        ("Depth Absolute Error", np.abs(prism_depth - gt_depth), "magma"),
    ]

    for ax, (title, image, cmap) in zip(axes[0], depth_items):
        if "Error" in title:
            handle = ax.imshow(image, cmap=cmap, vmin=0, vmax=20)
        else:
            handle = ax.imshow(image, vmin=depth_min, vmax=depth_max)
        ax.set_title(title)
        ax.axis("off")
        if title in {"Depth PRISM", "Depth Absolute Error"}:
            fig.colorbar(handle, ax=ax, fraction=0.046, pad=0.04)

    intensity_items = [
        ("Intensity Ground Truth", gt_intensity),
        ("Intensity Raw", raw_intensity),
        ("Intensity PRISM", prism_intensity),
    ]
    for ax, (title, image) in zip(axes[1, :3], intensity_items):
        ax.imshow(image, cmap="Grays_r")
        ax.set_title(title)
        ax.axis("off")

    axes[1, 3].axis("off")
    fig.tight_layout()
    plt.show()
