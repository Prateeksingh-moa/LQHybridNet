"""Print software/hardware versions for the reproducibility statement."""
import platform

import numpy
import pandas
import pennylane
import scipy
import sklearn
import timm
import torch
import torchvision

print(f"python {platform.python_version()}  ({platform.platform()})")
for m in (torch, torchvision, timm, pennylane, numpy, scipy, sklearn, pandas):
    print(f"{m.__name__} {m.__version__}")
print(f"CUDA {torch.version.cuda}  cuDNN {torch.backends.cudnn.version()}")
if torch.cuda.is_available():
    print("GPU", torch.cuda.get_device_name(0))
