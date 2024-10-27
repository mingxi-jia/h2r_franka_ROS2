# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import numpy as np
import torch

from src.skills.base_agent import base_agent

class OpenVLA(base_agent):
    def __init__(self, device='cpu'):
        super().__init__(device)
        # Load Processor & VLA
        self.sprocessor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
                "openvla/openvla-7b", 
                attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
                torch_dtype=torch.bfloat16, 
                low_cpu_mem_usage=True, 
                trust_remote_code=True
        ).to(device)

    def predict_action(self, image: Image.Image, instruction:str):
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        # Predict Action (7-DoF; un-normalize for BridgeData V2)
        inputs = self.processor(prompt, image).to("cpu", dtype=torch.bfloat16)
        action = self.vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        return action
