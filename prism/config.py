from dataclasses import dataclass


@dataclass(frozen=True)
class PrismConfig:
    data_path: str = "./data/spad_Middlebury_Art.mat"
    resolution: int = 512
    sdi_length: int = 128
    use_sdi_gating: bool = True
    sdi_margin: int = 30
    lir_window_size: int = 7
    nsr_clusters: int = 42
    rff_dim: int = 64
    rff_gamma: float = 0.5
    rff_projection_path: str = "./npy/Middlebury_128.npy"
    proxy_model_path: str = "./pth/Middlebury_128.pth"
    kmeans_model_path: str = "./model/Middlebury_128.model"
    depth_norm_bins: int = 1024
    plot_results: bool = True
    proxy_device: str = “cpu"