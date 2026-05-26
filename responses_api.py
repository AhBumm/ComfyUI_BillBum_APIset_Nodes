
import tenacity
import openai
from openai import OpenAI
import random
import base64
import json
from PIL import Image
import io
import torch
import numpy as np
import math
from comfy.utils import common_upscale


def downscale_input(image):
    samples = image.movedim(-1,1)
    #downscaling input images to roughly the same size as the outputs
    total = int(1536 * 1024)
    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
    if scale_by >= 1:
        return image
    width = round(samples.shape[3] * scale_by)
    height = round(samples.shape[2] * scale_by)

    s = common_upscale(samples, width, height, "lanczos", "disabled")
    s = s.movedim(1,-1)
    return s


class BillBum_Modified_Responses_API_Node:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text prompt for the Responses API",
                }),
                "model": ("STRING", {
                    "default": "gpt-4o",
                    "tooltip": "Model ID, e.g. gpt-4o, gpt-5.5, o3",
                }),
                "api_url": ("STRING", {
                    "multiline": False,
                    "default": "https://api.openai.com/v1",
                    "tooltip": "OpenAI-compatible API base URL",
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "YOUR_API_KEY_HERE",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                }),
                "temperature": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "-1 = none (use model default); 0~2 = custom temperature",
                }),
                "instructions": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "System/developer instructions inserted into the model context",
                }),
                "enable_image_generation": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable the image_generation tool for GPT Image generation/editing",
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Input image(s) for vision or image editing/reference. Supports batch input (multiple images).",
                }),
                "mask": ("MASK", {
                    "tooltip": "Optional mask for image editing (areas to edit)",
                }),
                "previous_response_id": ("STRING", {
                    "default": "",
                    "tooltip": "Previous response ID for multi-turn conversation",
                }),
                "reasoning_effort": (["none", "low", "medium", "high"], {
                    "default": "none",
                    "tooltip": "Reasoning effort for o-series/gpt-5 models (none = disabled)",
                }),
                "image_gen_options": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Optional JSON to override defaults for the image_generation tool. "
                        "Leave empty to use API defaults. Example:\n"
                        "  {\"quality\":\"high\",\"size\":\"1024x1536\","
                        "\"background\":\"transparent\",\"output_format\":\"png\"}\n"
                        "Supported keys: quality, size, background, output_format, "
                        "output_compression, n, input_fidelity, moderation. "
                        "The 'type' field cannot be overridden."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("STRING","IMAGE","STRING","STRING",)
    RETURN_NAMES = ("text_output","images","response_id","revised_prompt",)
    FUNCTION = "call_responses_api"
    CATEGORY = "BillBum_API"

    def _image_to_b64_url(self, single_image):
        image_np = (single_image.numpy() * 255).astype(np.uint8)
        img = Image.fromarray(image_np)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        base64_image = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{base64_image}"

    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1.25, min=5, max=30), stop=tenacity.stop_after_attempt(3), retry=tenacity.retry_if_exception_type((openai.APIConnectionError, openai.RateLimitError, openai.APITimeoutError, openai.InternalServerError)), reraise=True)
    def call_responses_api(
        self, prompt, model, api_url, api_key, seed, temperature,
        instructions, enable_image_generation,
        image=None, mask=None, previous_response_id=None,
        reasoning_effort="none",
        image_gen_options="",
    ):
        random.seed(seed)

        client = OpenAI(
            api_key=api_key,
            base_url=api_url,
        )

        # --- Build input content items ---
        content_items = []
        content_items.append({"type": "input_text", "text": prompt})

        # Append each input image from batch as base64
        if image is not None:
            batch_size = image.shape[0]
            for i in range(batch_size):
                scaled = downscale_input(image[i:i+1])  # keep batch dim for downscale
                b64_url = self._image_to_b64_url(scaled.squeeze(0))
                content_items.append({
                    "type": "input_image",
                    "image_url": b64_url,
                })

        # Build input
        input_data = [
            {
                "role": "user",
                "content": content_items,
            }
        ]

        # --- Build tools ---
        tools = []
        if enable_image_generation:
            image_gen_tool = {"type": "image_generation"}

            # Merge optional JSON overrides (advanced users only)
            if image_gen_options and image_gen_options.strip():
                try:
                    overrides = json.loads(image_gen_options)
                    if isinstance(overrides, dict):
                        overrides.pop("type", None)  # protect the 'type' field
                        image_gen_tool.update(overrides)
                    else:
                        print("[Responses API] image_gen_options is not a JSON object, ignored.")
                except json.JSONDecodeError as e:
                    print(f"[Responses API] Failed to parse image_gen_options as JSON: {e}. Ignored.")

            # Handle mask input (only meaningful when an input image is also given)
            if mask is not None and image is not None:
                # Resize mask to match the first input image's downscaled dimensions,
                # otherwise the mask won't align with the image the model edits.
                first_scaled = downscale_input(image[0:1])  # (1, H, W, C)
                target_h = int(first_scaled.shape[1])
                target_w = int(first_scaled.shape[2])

                m = mask[0].cpu()  # (H, W) — take first mask explicitly
                m_4d = m.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                if (m_4d.shape[2], m_4d.shape[3]) != (target_h, target_w):
                    m_4d = common_upscale(m_4d, target_w, target_h, "bilinear", "disabled")
                m_resized = m_4d.squeeze(0).squeeze(0)  # (target_h, target_w)

                rgba_mask = torch.zeros(target_h, target_w, 4, device="cpu")
                rgba_mask[:, :, 3] = (1 - m_resized).clamp(0.0, 1.0)

                mask_np = (rgba_mask.numpy() * 255).astype(np.uint8)
                mask_img = Image.fromarray(mask_np)
                mask_buf = io.BytesIO()
                mask_img.save(mask_buf, format="PNG")
                mask_b64 = base64.b64encode(mask_buf.getvalue()).decode('utf-8')
                image_gen_tool["input_image_mask"] = {
                    "image_url": f"data:image/png;base64,{mask_b64}",
                }
            elif mask is not None and image is None:
                print("[Responses API] mask provided without image; mask ignored.")

            tools.append(image_gen_tool)

        # --- Build request kwargs ---
        kwargs = {
            "model": model,
            "input": input_data,
        }

        # Only send temperature if not -1 (none = let API use model default)
        if temperature >= 0:
            kwargs["temperature"] = temperature

        if instructions.strip():
            kwargs["instructions"] = instructions

        if tools:
            kwargs["tools"] = tools

        if previous_response_id and previous_response_id.strip():
            kwargs["previous_response_id"] = previous_response_id.strip()

        if reasoning_effort != "none":
            kwargs["reasoning"] = {"effort": reasoning_effort}

        # --- API call ---
        print(f"[Responses API] Calling model={model}, tools={[t['type'] for t in tools]}")
        response = client.responses.create(**kwargs)
        print(f"[Responses API] Response ID: {response.id}, Status: {response.status}")

        # --- Process response outputs ---
        text_output = ""
        image_tensors = []
        revised_prompts = []
        response_id = response.id if hasattr(response, 'id') else ""

        for output_item in response.output:
            if output_item.type == "message":
                for content in output_item.content:
                    if hasattr(content, 'text') and content.text:
                        text_output += content.text
            elif output_item.type == "image_generation_call":
                if hasattr(output_item, 'result') and output_item.result:
                    img_data = base64.b64decode(output_item.result)
                    img = Image.open(io.BytesIO(img_data)).convert("RGBA")
                    img_array = np.array(img).astype(np.float32) / 255.0
                    img_tensor = torch.from_numpy(img_array)
                    image_tensors.append(img_tensor)
                if hasattr(output_item, 'revised_prompt') and output_item.revised_prompt:
                    revised_prompts.append(output_item.revised_prompt)

        # Fallback to output_text if no text found in message items
        if not text_output and hasattr(response, 'output_text') and response.output_text:
            text_output = response.output_text

        revised_prompt = "\n---\n".join(revised_prompts) if revised_prompts else ""

        # Stack image tensors or return a 1x1 placeholder
        if image_tensors:
            images = torch.stack(image_tensors, dim=0)
        else:
            images = torch.zeros(1, 1, 1, 4)

        return (text_output, images, response_id, revised_prompt)