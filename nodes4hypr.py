import torch
import numpy as np
from PIL import Image
import io
import base64
import requests
import random
import tenacity
import math
from comfy.utils import common_upscale


## ====== Utility Functions ======
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def downscale_input(image):

    samples = image.movedim(-1,1)
    total = int(1536 * 1024)
    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
    if scale_by >= 1:
        return image
    width = round(samples.shape[3] * scale_by)
    height = round(samples.shape[2] * scale_by)
    s = common_upscale(samples, width, height, "lanczos", "disabled")
    s = s.movedim(1,-1)
    return s


## ====== HyprLab API Nodes ======
class HyprLab_Image_API_Node:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True, "dynamicPrompts": True, "tooltip": "The main text prompt"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "model": ("STRING", {"default": "nano-banana-pro"}),
                "api_url": ("STRING", {"multiline": False, "default": "https://api.hyprlab.io/v1/images/generations"}),
                "api_key": ("STRING", {"multiline": False, "default": "YOUR_API_KEY_HERE"}),
                "resolution": (["1K", "2K", "4K"], {"default": "1K"}),
                "aspect_ratio": ([
                    "match_input_image", "1:1", "9:16", "16:9", "3:4", 
                    "4:3", "3:2", "2:3", "5:4", "4:5", "21:9"
                ], {"default": "1:1"}),
            },
            "optional": {
                "image_input": ("IMAGE", {"default": None, "tooltip": "Optional input images to guide generation"}),
            }
        }


    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "BillBum_API/Image_API"


    @staticmethod
    def _encode_images_to_base64(images):
        if images is None:
            return []
        if images.dim() == 3:
            images = images.unsqueeze(0)
        
        images = downscale_input(images)
        encoded_images = []
        for idx in range(images.shape[0]):
            tensor_image = images[idx].clamp(0.0, 1.0)
            
            pil_image = tensor2pil(tensor_image)
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            encoded_images.append(f"data:image/png;base64,{encoded}")
        return encoded_images

    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1.25, min=5, max=30), stop=tenacity.stop_after_attempt(3), retry=tenacity.retry_if_exception_type((requests.exceptions.ConnectionError, requests.exceptions.Timeout)), reraise=True)
    def generate_image(self, prompt, seed, model, api_url, api_key, resolution, aspect_ratio, image_input=None):
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": model,
            "prompt": prompt,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "output_format": "png",
            "response_format": "b64_json",
            "seed": seed
        }

        if image_input is not None:
            encoded_imgs = self._encode_images_to_base64(image_input)
            if encoded_imgs:
                payload["image_input"] = encoded_imgs

        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            response_data = response.json()
            
            images_output = []
            
            data_list = response_data.get("data", [])
            if not data_list and "b64_json" in response_data:
                data_list = [response_data]
            
            for item in data_list:
                b64_str = item.get("b64_json")
                if b64_str:
                    img_data = base64.b64decode(b64_str)
                    img = Image.open(io.BytesIO(img_data))
                    
                    if img.mode != "RGBA":
                        img = img.convert("RGBA")
                    
                    images_output.append(pil2tensor(img))
            
            if not images_output:
                print(f"API Response: {response_data}")
                raise ValueError("API did not return any valid images.")

            return (torch.cat(images_output, dim=0),)

        except Exception as e:
            if isinstance(e, requests.exceptions.RequestException) and e.response is not None:
                print(f"API Error Response: {e.response.text}")
            raise ValueError(f"HyprBanana API Error: {e}")

