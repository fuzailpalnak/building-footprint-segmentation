"""Verify PyTorch can see the local GPU."""

from building_footprint_segmentation._env import configure_windows_openmp

configure_windows_openmp()

import torch

from building_footprint_segmentation.utils.py_network import get_device


def _print_cpu_torch_fix():
    print()
    print("You have the CPU-only PyTorch build. Reinstall the CUDA build:")
    print()
    print("  pip uninstall -y torch torchvision torchaudio")
    print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
    print()
    print("Or with conda:")
    print()
    print("  conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia")
    print()


def main():
    device = get_device()
    version = torch.__version__
    print(f"PyTorch version: {version}")
    print(f"Selected device: {device}")

    if "+cpu" in version:
        print("Build type: CPU (no GPU support)")
        _print_cpu_torch_fix()
        return

    if device.type == "cuda":
        print("CUDA available: True")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
    else:
        print("CUDA available: False")
        print("NVIDIA driver is present but this PyTorch build has no CUDA.")
        _print_cpu_torch_fix()


if __name__ == "__main__":
    main()
