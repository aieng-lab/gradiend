from typing import Tuple

import torch


def cuda_unusable_runtime_error(cuda_count: int) -> RuntimeError:
    torch_cuda_version = getattr(torch.version, "cuda", None) or "unknown"
    return RuntimeError(
        "CUDA devices are visible but PyTorch cannot initialize CUDA. "
        f"torch.cuda.is_available() is False while torch.cuda.device_count() is {cuda_count}. "
        f"Installed torch reports CUDA runtime {torch_cuda_version}. "
        "This usually means the NVIDIA driver is too old for the installed PyTorch CUDA build, "
        "or CUDA is otherwise unusable in the current container. Install a PyTorch build compatible "
        "with the host driver, update the NVIDIA driver, or pass device='cpu' to run intentionally on CPU."
    )


def validate_cuda_usable_if_visible(*, allow_unusable_visible_cuda: bool = False) -> Tuple[bool, int]:
    """Return CUDA availability/count, raising when visible devices cannot be initialized.

    Set allow_unusable_visible_cuda=True only for callers that intentionally force CPU.
    """
    cuda_available = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count()
    if not cuda_available and cuda_count > 0 and not allow_unusable_visible_cuda:
        raise cuda_unusable_runtime_error(cuda_count)
    return cuda_available, cuda_count

