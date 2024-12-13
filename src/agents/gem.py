# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import numpy as np
import torch


class GEM():
    def __init__(self, device='cpu'):
        super().__init__(device)
        pass

    def predict_action(self, image: Image.Image, instruction:str):
        pass
