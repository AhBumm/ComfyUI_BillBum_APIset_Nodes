import io
from PIL import Image
import numpy as np
import torch
import requests
import base64
import time
import json
import tenacity
import math
from comfy.utils import common_upscale


## DataType Conversion Functions
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2ndarray(image):
    return np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)

def ndarray2tensor(image):
    return torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)


## Node Classes
class seedance_api_node:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("STRING", {"default": "doubao-seedance-1-5-pro-251215"}),
                "prompt": ("STRING", {"forceInput": True}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffff}),
                "api_url": ("STRING", {"default": "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks"}),
                "api_key": ("STRING", {"default": "Input_your_API_key_here..."}),
                "resolution": (["480p", "720p", "1080p"], {"default":"480p"}),
                "ratio": (["16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive"], {"default":"adaptive"}),
                "duration": ("INT", {"default":5, "min":1, "max":12, "step":1}),
                "camerafixed": (["true", "false"], {"default":"false"}),
                "watermark": (["true", "false"], {"default":"false"})
            },
            "optional": {
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response_str",)
    FUNCTION = "create_seedance_task"
    CATEGORY = "BillBum/API Nodes"

    def _poll_task_status(self, task_id, api_url, api_key, interval=1, max_attempts=500):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        url = f"{api_url}/{task_id}"

        for attempt in range(max_attempts):
            try:
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    task_data = response.json()
                    status = task_data.get("status")
                    response_text = json.dumps(task_data, indent=2, ensure_ascii=False)
                    if status in ["succeeded", "failed", "cancelled"]:
                        return response_text
                    else:
                        time.sleep(interval)
                else:
                    return f"Failed to fetch task status. HTTP Status Code: {response.status_code}\nResponse: {response.text}"
            except Exception as e:
                return f"An exception occurred: {str(e)}"
        return "Polling timed out."

    def _to_base64_url(self, image_tensor):
        pil_image = tensor2pil(image_tensor)
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"

    def create_seedance_task(self, model, prompt, seed, api_url, api_key, resolution, ratio, duration, camerafixed, watermark, first_frame=None, last_frame=None):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        content = [{"type": "text", "text": prompt}]

        if first_frame is not None:
            fsf_b64url = self._to_base64_url(first_frame)
            content.append({
                "type": "image_url",
                "image_url": {"url": fsf_b64url},
                "role": "first_frame"
            })

        if last_frame is not None:
            lsf_b64url = self._to_base64_url(last_frame)
            content.append({
                "type": "image_url",
                "image_url": {"url": lsf_b64url},
                "role": "last_frame"
            })

        data = {
            "model": model,
            "content": content,
            "ratio": ratio,
            "resolution": resolution,
            "camera_fixed": True if camerafixed == "true" else False,
            "watermark": True if watermark == "true" else False
        }

        # 检查是否为文生视频 (T2V) 模式
        is_t2v = all(item.get("type") == "text" for item in content)
        
        # 针对 1.5-pro 系列模型，经过测试传参 duration（无论在 body 还是 prompt 中）均会导致 400 错误
        # 官方 1.5 模型目前可能为固定时长，故直接忽略该参数以确保调用成功
        if "doubao-seedance-1-5-pro" in model:
            data["generate_audio"] = True
            # 不发送 duration 参数
        else:
            # 1.0 等旧版模型仍需发送 duration
            data["duration"] = duration
            
        if seed != -1:
            data["seed"] = seed
        
        try:
            response = requests.post(api_url, headers=headers, json=data)
            if response.status_code == 200:
                response_json = response.json()
                task_id = response_json.get("id", "")
                if task_id:
                    return (self._poll_task_status(task_id, api_url, api_key),)
                else:
                    return (f"Task ID not found. Response: {response.text}",)
            else:
                return (f"Failed to create task. HTTP {response.status_code}\nResponse: {response.text}",)
        except Exception as e:
            return (f"An exception occurred: {str(e)}",)


class seedream_api_node:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("STRING", {"default": "doubao-seedream-4-0-250828"}),
                "prompt": ("STRING", {"forceInput": True}),
                "size": (["1K","2K","4K"], {"default": "1K"}),
                "api_url": ("STRING", {"default": "https://ark.cn-beijing.volces.com/api/v3/images/generations"}),
                "api_key": ("STRING", {"default": "Input_your_API_key_here..."}),
                "story_mode": (
                    ["disabled", "auto"],
                    {
                        "default": "disabled",
                        "description": "Enable Story Mode for generating images with consistent elements across multiple generations. (Only for Seedream 4.0 and later models)"
                    },
                ),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "response_str")
    FUNCTION = "generate_image"
    CATEGORY = "BillBum/API Nodes"

    def _to_base64_url_from_input(self, img_input):
        # Accept torch tensor (single or batch), PIL Image or numpy array
        # If torch tensor batch: iterate and return list of data URLs
        urls = []
        if isinstance(img_input, torch.Tensor):
            imgs = img_input
            if imgs.dim() == 3:
                imgs = imgs.unsqueeze(0)

            # downscale large inputs to a reasonable size
            samples = imgs.movedim(-1, 1)
            total = int(1536 * 1024)
            scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
            if scale_by < 1:
                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)
                s = common_upscale(samples, width, height, "lanczos", "disabled")
                imgs = s.movedim(1, -1)

            for idx in range(imgs.shape[0]):
                pil_image = tensor2pil(imgs[idx])
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                urls.append(f"data:image/png;base64,{img_str}")
            return urls

        # single PIL or numpy image
        if isinstance(img_input, Image.Image):
            pil_image = img_input
        else:
            try:
                pil_image = Image.fromarray(np.array(img_input))
            except Exception:
                raise TypeError("Unsupported IMAGE input type")

        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return [f"data:image/png;base64,{img_str}"]

    def _decode_b64_to_tensor(self, b64_string):
        if b64_string.startswith(("data:image/png;base64,", "data:image/jpeg;base64,", "data:image/webp;base64,")):
            b64 = b64_string.split(",", 1)[1]
        else:
            b64 = b64_string
        image_data = base64.b64decode(b64)
        image = Image.open(io.BytesIO(image_data))
        return pil2tensor(image)

    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1.25, min=2, max=20), stop=tenacity.stop_after_attempt(3), reraise=True)
    def generate_image(self, model, prompt, size, api_url, api_key, story_mode, image=None):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        if image is not None and model.startswith("doubao-seedream-4-5") and size == "1K":
            size = "2K"

        payload = {
            "model": model,
            "prompt": prompt,
            "response_format": "b64_json",
            "size": size,
            "stream": False,
            "watermark": False,
        }
        
        if story_mode == "disabled" and model.startswith("doubao-seedream-4"):
            payload["sequential_image_generation"] = "disabled"

        if story_mode == "auto" and model.startswith("doubao-seedream-4"):
            payload["sequential_image_generation"] = "auto"

        encoded_imgs = []
        if image is not None:
            try:
                encoded_imgs = self._to_base64_url_from_input(image)
            except Exception as e:
                return (None, f"Error encoding input image: {e}")

        if encoded_imgs:
            if len(encoded_imgs) == 1:
                payload["image"] = encoded_imgs[0]
            else:
                payload["image"] = encoded_imgs

        try:
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
        except Exception as e:
            debug_info = {
                "request_payload": payload,
                "error_message": str(e),
                "response_text": getattr(response, 'text', 'No response text available')
            }
            pretty_debug = json.dumps(debug_info, indent=2, ensure_ascii=False)
            return (None, f"Request failed:\n{pretty_debug}")

        try:
            response_json = response.json()
        except Exception as e:
            debug_info = {
                "request_payload": payload,
                "error_message": str(e),
                "raw_response": getattr(response,'text',str(response))
            }
            pretty_debug = json.dumps(debug_info, indent=2, ensure_ascii=False)
            return (None, f"Failed to parse JSON response:\n{pretty_debug}")

        images_output = []
        data_list = response_json.get("data", [])
        if not data_list and "b64_json" in response_json:
            data_list = [response_json]

        for item in data_list:
            b64_str = item.get("b64_json")
            if not b64_str:
                continue
            try:
                img_tensor = self._decode_b64_to_tensor(b64_str)
                images_output.append(img_tensor)
            except Exception as e:
                print(f"Failed decoding an image from response: {e}")

        debug_info = {
            "request_payload": payload,
            "response": response_json
        }
        pretty = json.dumps(debug_info, indent=2, ensure_ascii=False)

        if not images_output:
            return (None, f"Unexpected or empty image response:\n{pretty}")

        try:
            batch = torch.cat(images_output, dim=0)
        except Exception:
            batch = images_output[0]

        return (batch, pretty)

