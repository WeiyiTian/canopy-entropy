import gc
import torch


def clear_runtime_memory() -> None:
    """Run Python and CUDA cleanup after large model workloads."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
