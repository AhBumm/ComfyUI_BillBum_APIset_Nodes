import tenacity
from openai import OpenAI
import random
import base64
from PIL import Image
import os
import io
import torch
import numpy as np
from torchvision.transforms.v2 import ToPILImage
import tempfile
import requests
import json
from together import Together
import re

META_PROMPT = """
Given a task description or existing prompt, produce a detailed system prompt to guide a language model in completing the task effectively.

# Guidelines

- Understand the Task: Grasp the main objective, goals, requirements, constraints, and expected output.
- Minimal Changes: If an existing prompt is provided, improve it only if it's simple. For complex prompts, enhance clarity and add missing elements without altering the original structure.
- Reasoning Before Conclusions**: Encourage reasoning steps before any conclusions are reached. ATTENTION! If the user provides examples where the reasoning happens afterward, REVERSE the order! NEVER START EXAMPLES WITH CONCLUSIONS!
    - Reasoning Order: Call out reasoning portions of the prompt and conclusion parts (specific fields by name). For each, determine the ORDER in which this is done, and whether it needs to be reversed.
    - Conclusion, classifications, or results should ALWAYS appear last.
- Examples: Include high-quality examples if helpful, using placeholders [in brackets] for complex elements.
   - What kinds of examples may need to be included, how many, and whether they are complex enough to benefit from placeholders.
- Clarity and Conciseness: Use clear, specific language. Avoid unnecessary instructions or bland statements.
- Formatting: Use markdown features for readability. DO NOT USE ``` CODE BLOCKS UNLESS SPECIFICALLY REQUESTED.
- Preserve User Content: If the input task or prompt includes extensive guidelines or examples, preserve them entirely, or as closely as possible. If they are vague, consider breaking down into sub-steps. Keep any details, guidelines, examples, variables, or placeholders provided by the user.
- Constants: DO include constants in the prompt, as they are not susceptible to prompt injection. Such as guides, rubrics, and examples.
- Output Format: Explicitly the most appropriate output format, in detail. This should include length and syntax (e.g. short sentence, paragraph, JSON, etc.)
    - For tasks outputting well-defined or structured data (classification, JSON, etc.) bias toward outputting a JSON.
    - JSON should never be wrapped in code blocks (```) unless explicitly requested.

The final prompt you output should adhere to the following structure below. Do not include any additional commentary, only output the completed system prompt. SPECIFICALLY, do not include any additional messages at the start or end of the prompt. (e.g. no "---")

[Concise instruction describing the task - this should be the first line in the prompt, no section header]

[Additional details as needed.]

[Optional sections with headings or bullet points for detailed steps.]

# Steps [optional]

[optional: a detailed breakdown of the steps necessary to accomplish the task]

# Output Format

[Specifically call out how the output should be formatted, be it response length, structure e.g. JSON, markdown, etc]

# Examples [optional]

[Optional: 1-3 well-defined examples with placeholders if necessary. Clearly mark where examples start and end, and what the input and output are. User placeholders as necessary.]
[If the examples are shorter than what a realistic example is expected to be, make a reference with () explaining how real examples should be longer / shorter / different. AND USE PLACEHOLDERS! ]

# Notes [optional]

[optional: edge cases, details, and an area to call or repeat out specific important considerations]
""".strip()

def pil2tensor(images: Image.Image | list[Image.Image]) -> torch.Tensor:
    """Converts a PIL Image or a list of PIL Images to a tensor."""

    def single_pil2tensor(image: Image.Image) -> torch.Tensor:
        np_image = np.array(image).astype(np.float32) / 255.0
        if np_image.ndim == 2:  # Grayscale
            return torch.from_numpy(np_image).unsqueeze(0)  # (1, H, W)
        else:  # RGB or RGBA
            return torch.from_numpy(np_image).unsqueeze(0)  # (1, H, W, C)

    if isinstance(images, Image.Image):
        return single_pil2tensor(images)
    else:
        return torch.cat([single_pil2tensor(img) for img in images], dim=0)

class BillBum_Modified_Dalle_API_Node:

    def __init__(self):
            pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True},),
                "size": (["1024x1024", "512x512", "256x256"],),
                "quality": (["hd", "standard"],),
                "n": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                    "step": 1,
                    "display": "number",
                }),
                "api_url": ("STRING", {
                    "multiline": False,
                    "default": "https://api.hyprlab.io/v1",
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "YOUR_API_KEY_HERE",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("image_url",)
    FUNCTION = "get_dalle_3_image"
    CATEGORY = "BillBum_API"

    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10))
    def get_dalle_3_image(self, prompt, size, quality, n, api_url, api_key):
        client = OpenAI(
            api_key=api_key,
            base_url=api_url
        )
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality=quality,
            n=n,
        )

        image_url = response.data[0].url
        return (image_url,)

class BillBum_Modified_LLM_API_Node:

    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {
                     "default": 0, "min": 0, "max": 0xffffffffffffffff
                }),
                "prompt": ("STRING", {"forceInput": True},),
                "model": ("STRING", {
                    "default": "gpt-4o",
                }),
                "api_url": ("STRING", {
                    "multiline": False,
                    "default": "https://api.hyprlab.io/v2",
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "YOUR_API_KEY_HERE",
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                }),
                "use_meta_prompt": ("BOOLEAN", {
                    "default": False,
                }),
            },
        }

    RETURN_TYPES = ("STRING","INT","STRING","STRING","STRING",)
    RETURN_NAMES = ("LLM ANSWERS","seed","model","api_url","api_key",)
    FUNCTION = "get_llm_response"
    CATEGORY = "BillBum_API"

    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10))
    def get_llm_response(self, prompt, model, api_url, api_key, system_prompt, use_meta_prompt, seed):

        random.seed(seed)
        final_system_prompt = META_PROMPT if use_meta_prompt else system_prompt

        client = OpenAI(
            api_key=api_key,
            base_url=api_url
        )
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {'role':'system', 'content': final_system_prompt},
                {'role': 'user', 'content': prompt}]
        )
        return (completion.choices[0].message.content,seed,)

class BillBum_Modified_Structured_LLM_Node:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True},),
                "model": ("STRING", {
                    "default": "gpt-4o-mini",
                }),
                "api_url": ("STRING", {
                    "multiline": False,
                    "default": "https://api.hyprlab.io/v1",
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "YOUR_API_KEY_HERE",
                }),
                "system_prompt": ("STRING", {"forceInput": True},),
                "output_format": ("STRING", {"forceInput": True},),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("structured_str",)
    FUNCTION = "get_llm_structured_response"
    CATEGORY = "BillBum_API"

    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10))
    def get_llm_structured_response(self, prompt, model, api_url, api_key, system_prompt, output_format, seed):
        
        META_SCHEMA = json.loads(output_format)

        random.seed(seed)

        client = OpenAI(
            api_key=api_key,
            base_url=api_url
        )

        completion = client.chat.completions.create(
            model=model,
            response_format={"type": "json_schema", "json_schema": META_SCHEMA},
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': "Description:\n" + prompt}
            ]
        )

        response_dict = json.loads(completion.choices[0].message.content)
        data = response_dict.get("data", [])
        return (data,)

class BillBum_Modified_LLM_API_sequentialNode:

    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True},),
                "model": ("STRING", {"forceInput": True},),
                "api_url": ("STRING", {"forceInput": True},),
                "api_key": ("STRING", {"forceInput": True},),
                "system_prompt": ("STRING", {
                    "multiline": True,
                }),
            },
        }

    RETURN_TYPES = ("STRING","STRING","STRING","STRING",)
    RETURN_NAMES = ("LLM ANSWERS","model","api_url","api_key",)
    FUNCTION = "get_llm_response_sequential"
    CATEGORY = "BillBum_API"

    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10))
    def get_llm_response_sequential(self, prompt, model, api_url, api_key, system_prompt, seed):
        
        random.seed(seed)
        client = OpenAI(
            api_key=api_key,
            base_url=api_url
        )
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {'role':'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}]
        )
        return (completion.choices[0].message.content,seed,)

class BillBum_Modified_VisionLM_API_Node:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff
                }),
                "prompt": ("STRING", {"forceInput": True},),
                "model": ("STRING", {
                    "default": "qwen-vl-max-0201",
                }),
                "api_url": ("STRING", {
                    "multiline": False,
                    "default": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "YOUR_API_KEY_HERE",
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                }),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING","INT","STRING","STRING","STRING",)
    RETURN_NAMES = ("LLM ANSWERS","seed","model","api_url","api_key",)
    FUNCTION = "get_vlm_response"
    CATEGORY = "BillBum_API"

    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10))
    def get_vlm_response(self, prompt, model, api_url, api_key, system_prompt, image, seed):

        random.seed(seed)

        client = OpenAI(
            api_key=api_key,
            base_url=api_url
        )
        with torch.no_grad():
            pil_image = ToPILImage()(image.permute([0, 3, 1, 2])[0]).convert("RGB")
        image_path = tempfile.NamedTemporaryFile(suffix=".png").name
        pil_image.save(image_path)

        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        completion = client.chat.completions.create(
            model=model,
            messages=[
                {'role':'system', 'content': system_prompt},
                {'role': 'user', 'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ]
        )
        return (completion.choices[0].message.content, seed, model, api_url, api_key)

class BillBum_Modified_img2url_Node:
    """
    A ComfyUI node to convert an image file to a base64 encoded string.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("base64_url",)
    FUNCTION = "get_base64_image"
    CATEGORY = "BillBum Image Processing"

    def get_base64_image(self, image):
        with torch.no_grad():
            pil_image = ToPILImage()(image.permute([0, 3, 1, 2])[0]).convert("RGB")
        image_path = tempfile.NamedTemporaryFile(suffix=".png").name
        pil_image.save(image_path)

        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            image_url = f"data:image/jpeg;base64,{base64_image}"

            return (image_url,)

        finally:
            os.remove(image_path)

class BillBum_Modified_Ideogram_API_Node:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True},),
                "negative_prompt": ("STRING", {"forceInput": True},),
                "model": (["V_2", "V_2_TURBO", "V_1", "V_1_TURBO"],),
                "magic_prompt_option": (["AUTO", "ON", "OFF"],),
                "aspect_ratio": (["ASPECT_1_1", "ASPECT_1_3", "ASPECT_3_1", "ASPECT_16_10", "ASPECT_10_16", "ASPECT_4_3", "ASPECT_3_4", "ASPECT_9_16", "ASPECT_16_9", "ASPECT_2_3", "ASPECT_3_2"],),
                "seed": ("INT", {
                     "default": 0, "min": 0, "max": 0xffffffffffffffff
                }),
                "api_url": ("STRING", {
                    "multiline": False,
                    "default": "https://api.ideogram.ai/generate",
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "YOUR_API_KEY_HERE",
                }),
                "style_type": (
                    ["None", "GENERAL", "REALISTIC", "DESIGN", "RENDER_3D", "ANIME"],
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("image_url",)
    FUNCTION = "get_ideogram_image"
    CATEGORY = "BillBum_API"

    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10))
    def get_ideogram_image(self, prompt, negative_prompt, aspect_ratio, model, magic_prompt_option, seed, api_url, api_key, style_type,):

        random.seed(seed)
        url = api_url
        
        payload = { "image_request": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "aspect_ratio": aspect_ratio,
                "model": model,
                "magic_prompt_option": magic_prompt_option,
                "seed": seed
            } }
        
        if model in ["V_2", "V_2_TURBO"] and style_type != "None":
            payload["image_request"]["style_type"] = style_type

        headers = {
            "Api-Key": api_key,
            "Content-Type": "application/json"
        }

        response = requests.post(url, json=payload, headers=headers)
        image_url = response.json()["data"][0]["url"]
        return (image_url,)

class BillBum_Modified_Text2Image_API_Node:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("STRING", {"default": "flux-1.1-pro"}),
                "prompt": ("STRING", {"forceInput": True}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 1440, "step": 32, "display": "number"}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 1440, "step": 32, "display": "number"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 50, "step": 1, "display": "number"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "api_url": ("STRING", {"multiline": False, "default": "https://api.hyprlab.io/v1/images/generations"}),
                "api_key": ("STRING", {"default": "YOUR_API_KEY_HERE"}),
            },
        }

    RETURN_TYPES = ("STRING", "INT",)
    RETURN_NAMES = ("image_url", "seed",)
    FUNCTION = "get_t2i_image"
    CATEGORY = "BillBum_API"

    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10))
    def get_t2i_image(self, model, prompt, width, height, steps, api_url, seed, api_key):
        random.seed(seed)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        data = {
            "model": model,
            "prompt": prompt,
            "steps": steps,
            "height": height,
            "width": width,
            "response_format": "url",
            "output_format": "webp"
        }

        response = requests.post(api_url, headers=headers, json=data)
        response_data = response.json()
        
        image_url = response_data.get("data", [{}])[0].get("url")
        if not image_url:
            image_url = "https://external-preview.redd.it/1RUrKk4LhgVbp_Z4JwbAPE4C0ZMxQNw0ueHlTFoykcc.jpg?auto=webp&s=ef14b6dc9fed780158b0842145f5fd017ffac98f"
        return (image_url, seed)

class BillBum_Modified_Together_API_Node:

    def __init__(self):
            pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("STRING", {
                    "default": "black-forest-labs/FLUX.1.1-pro",
                }),
                "prompt": ("STRING", {
                    "forceInput": True,
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 2048,
                    "step": 64,
                    "display": "number",
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 2048,
                    "step": 64,
                    "display": "number",
                }),
                "steps": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 30,
                    "step": 1,
                    "display": "number",
                }),
                "n": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                    "step": 1,
                    "display": "number",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                }),
                "api_key": ("STRING", {
                    "default": "YOUR_API_KEY_HERE",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Base64_url",)
    FUNCTION = "get_together_image"
    CATEGORY = "BillBum_API"

    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10))
    def get_together_image(self, model, prompt, width, height, steps, n, seed, api_key):

        random.seed(seed)

        client = Together(
            api_key=api_key,
        )
        response = client.images.generate(
            prompt=prompt,
            model=model,
            width=width,
            height=height,
            steps=steps,
            n=n,
            seed=seed,
            response_format="b64_json",
        )
        b64_string = response.data[0].b64_json
        base64_url = f"data:image/png;base64,{b64_string}"
        return (base64_url,)

class BillBum_Modified_Base64_Url2Img_Node:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base64_url": ("STRING", {
                    "forceInput": True,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "get_base64_image"
    CATEGORY = "BillBum Image Processing"

    def get_base64_image(self, base64_url):
        base64_data = base64_url.split(",")[1]
        image_data = base64.b64decode(base64_data)
        image = Image.open(io.BytesIO(image_data))
        return (pil2tensor(image),)

class BillBum_Modified_ImageSplit_Node:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "split_image"
    CATEGORY = "BillBum Image Processing"

    def split_image(self, image):
        # Convert tensor to PIL image if necessary
        img_pil = Image.fromarray((image.squeeze(0).numpy() * 255).astype('uint8')) if isinstance(image, torch.Tensor) else image
        
        # Handle different image sizes
        if img_pil.size == (1024, 1024):
            # If the image is 1024x1024, return it as is
            return (image,)
        elif img_pil.size == (2048, 1024):
            # If the image is 2048x1024, split it into two 1024x1024 images
            box_coordinates = [(0, 0, 1024, 1024), (1024, 0, 2048, 1024)]
        elif img_pil.size == (2048, 2048):
            # If the image is 2048x2048, split it into four 1024x1024 images
            box_coordinates = [(0, 0, 1024, 1024), (1024, 0, 2048, 1024), (0, 1024, 1024, 2048), (1024, 1024, 2048, 2048)]
        else:
            raise ValueError("Input image must be either 1024x1024, 2048x1024, or 2048x2048.")
        
        # Define tolerance for detecting near-black images
        tolerance = 0.08  # 8% tolerance

        # Crop and convert the sub-images to tensors, while filtering out near-black images
        sub_images = []
        for box in box_coordinates:
            cropped_img = img_pil.crop(box)
            np_img = np.array(cropped_img).astype(np.float32) / 255.0
            
            # Check if the image is not near-black
            if not np.all(np_img <= tolerance):
                sub_images.append(torch.from_numpy(np_img))

        # If all sub-images are near-black, return an empty tensor
        if not sub_images:
            raise ValueError("All cropped sub-images are near-black.")

        # Stack valid sub-images to create a batch tensor
        batch_tensor = torch.stack(sub_images)

        return (batch_tensor,)
    
class BillBum_Modified_Base64_Url2Data_Node:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base64_url": ("STRING", {"forceInput": True},),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "convert"
    CATEGORY = "BillBum String Processing"

    def convert(self, base64_url_string):
        try:
            base64_data_string = base64_url_string.split(",", 1)[1]
        except IndexError:
            raise ValueError("Invalid base64 URL string format.")
        
        return (base64_data_string,)

class BillBum_Modified_RegText_Node:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_text": ("STRING", {"forceInput": True},),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("formatted_text",)
    FUNCTION = "de_warp_text"
    CATEGORY = "text_processing"

    def de_warp_text(self, input_text):
        # Remove newline characters and replace periods with commas
        text = input_text.replace('\n', '').replace('.', ',')
        
        # Use regex to remove all characters except letters, numbers, spaces, commas, single quotes, and hyphen
        text = re.sub(r"[^a-zA-Z0-9\s,'-]", "", text)
        
        return (text,)

# Ensure these mappings are correctly integrated into your ComfyUI environment
NODE_CLASS_MAPPINGS = {
}

NODE_DISPLAY_NAME_MAPPINGS = {
}