import tenacity
import random
from openai import OpenAI
import io
import re
from PIL import Image
import numpy as np
import torch
import requests
import math
import base64
from comfy.utils import common_upscale
import subprocess
import tempfile
import os
from urllib.parse import urlparse
import folder_paths
import shutil
from comfy_api.latest import ui
from comfy_api.latest import io as comfyio
from comfy_api.input_impl import VideoFromFile
from urlextract import URLExtract


## ======== Utils Functions ========
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

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


## ======== Nodes Classes ========
class BillBum_Modified_StreamResponse_LLM_API:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "model": ("STRING", {"default": "gpt-4o-mini"}),
                "api_url": ("STRING", {"multiline": False, "default": "https://api.tu-zi.com/v1"}),
                "api_key": ("STRING", {"multiline": False, "default": "YOUR_API_KEY_HERE"}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "enable_thinking": ("COMBO", {
                    "options": ["true", "false", "none"],
                    "default": "none",
                    "tooltip": "only true/false would append 'enable_thinking' to request body",
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {"forceInput": True, "default": None}),
                "images": ("IMAGE", {"default": None, "tooltip": "Use Any Image Batch Nodes to input multiple images"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("LLM RESPONSE",)
    FUNCTION = "get_llm_stream_response"
    CATEGORY = "BillBum_API/Stream Response"

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
            encoded_images.append(encoded)
        return encoded_images

    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1.25, min=5, max=30), stop=tenacity.stop_after_attempt(3), reraise=True)
    def get_llm_stream_response(
        self,
        prompt,
        seed,
        model,
        api_url,
        api_key,
        temperature,
        enable_thinking,
        images=None,
        system_prompt=None,
    ):
        random.seed(seed)
        client = OpenAI(api_key=api_key, base_url=api_url)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_content = []
        if prompt:
            user_content.append({"type": "text", "text": prompt})

        for encoded_image in self._encode_images_to_base64(images):
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
            })

        if not user_content:
            raise ValueError("Prompt and images cannot both be empty.")

        messages.append({"role": "user", "content": user_content})

        request_kwargs = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if temperature != 0.0:
            request_kwargs["temperature"] = temperature

        extra_body = {}
        if enable_thinking == "true":
            extra_body["enable_thinking"] = True
        elif enable_thinking == "false":
            extra_body["enable_thinking"] = False
        if extra_body:
            request_kwargs["extra_body"] = extra_body

        try:
            completion = client.chat.completions.create(**request_kwargs)

            full_content = ""
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    delta = chunk.choices[0].delta.content
                    full_content += delta
                    # print(delta, end="")  # For debugging stream output
            return (full_content,)
        
        except Exception as e:
            print(f"LLM API Error: {type(e).__name__} - {e}")
            # Re-raise the exception to allow tenacity to handle retries
            raise


class Url2Image:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "get_url_image"
    CATEGORY = "BillBum_API/Utils"

    def _load_image_bytes(self, entry: str) -> bytes:
        if entry.startswith("data:"):
            _, base64_data = entry.split(",", 1)
            return base64.b64decode(base64_data)
        if entry.startswith(("http://", "https://")):
            response = requests.get(entry, timeout=10)
            response.raise_for_status()
            return response.content
        return base64.b64decode(entry)

    def _decode_entry(self, entry: str):
        image_data = self._load_image_bytes(entry)
        image = Image.open(io.BytesIO(image_data))
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        return np.array(image, dtype=np.float32) / 255.0

    def get_url_image(self, url):
        if not url:
            return (None,)

        entries = []
        for raw_line in url.replace("\r", "").split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("data:"):
                entries.append(line)
            else:
                for part in line.split(","):
                    part = part.strip()
                    if part:
                        entries.append(part)

        if not entries:
            return (None,)

        decoded_images = []
        for entry in entries:
            try:
                decoded_images.append(self._decode_entry(entry))
            except Exception as e:
                print(f"Url2Image: Can't decode {entry}: {e}")

        if not decoded_images:
            return (None,)

        max_h = max(img.shape[0] for img in decoded_images)
        max_w = max(img.shape[1] for img in decoded_images)

        batches = []
        for img in decoded_images:
            h, w, _ = img.shape
            padded = np.zeros((max_h, max_w, 4), dtype=np.float32)
            padded[:h, :w, :] = img
            batches.append(padded)

        image_tensor = torch.from_numpy(np.stack(batches, axis=0))
        return (image_tensor,)


class LoadVideoFromUrlComfyIO(comfyio.ComfyNode):

    def __init__(self):
        pass

    @classmethod
    def define_schema(cls):
        return comfyio.Schema(
            node_id="load_video_from_url_comfy_core",
            display_name="Load&Save Video From URL (Comfy Core)",
            category="BillBum_API/Utils",
            inputs=[
                comfyio.String.Input("url", default="", tooltip="http/https url"),
                comfyio.String.Input("filename_prefix", default="video_files/url_download"),
            ],
            outputs=[comfyio.Video.Output("video")],
            hidden=[comfyio.Hidden.prompt, comfyio.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @staticmethod
    def _extension_from_url(url: str) -> str:
        ext = os.path.splitext(urlparse(url).path)[1].lower()
        if ext in {".mp4", ".mov", ".mkv", ".webm", ".gif"}:
            return ext
        return ".mp4"

    @staticmethod
    def _download_to_temp(url: str, suffix: str) -> str:
        with requests.get(url, stream=True, timeout=30) as resp:
            resp.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        tmp.write(chunk)
                return tmp.name

    @classmethod
    def execute(cls, url, filename_prefix) -> comfyio.NodeOutput:
        url = (url or "").strip()
        if not url:
            raise ValueError("URL cannot be empty.")

        suffix = cls._extension_from_url(url)
        temp_path = cls._download_to_temp(url, suffix)

        try:
            video_temp = VideoFromFile(temp_path)
            width, height = video_temp.get_dimensions()

            full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
                filename_prefix or "temp",
                folder_paths.get_output_directory(),
                width,
                height,
            )

            output_name = f"{filename}_{counter:05}{suffix}"
            final_path = os.path.join(full_output_folder, output_name)
            shutil.move(temp_path, final_path)

            video = VideoFromFile(final_path)
            preview = ui.PreviewVideo([ui.SavedResult(output_name, subfolder, comfyio.FolderType.output)])

            return comfyio.NodeOutput(video, ui=preview)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class LoadVideoFromUrlVHS:

    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "VHS_VIDEOINFO",)
    RETURN_NAMES = ("image", "video_info",)
    FUNCTION = "load_video"
    CATEGORY = "BillBum_API/Utils"

    def _download_video(self, url: str) -> str:
        if not url.startswith(("http://", "https://")):
            raise ValueError("仅支持 http/https URL。")
        try:
            with requests.get(url, timeout=15, stream=True) as response:
                response.raise_for_status()
                suffix = os.path.splitext(url)[1] or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            tmp_file.write(chunk)
                    return tmp_file.name
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"无法下载视频: {e}")

    def _get_video_metadata(self, filepath: str):
        ffmpeg_path = "ffmpeg"
        width = height = 0
        fps = 30.0
        duration = 0.0

        try:
            proc = subprocess.run(
                [ffmpeg_path, "-i", filepath, "-f", "null", "-"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
            )
            stderr_output = proc.stderr.decode("utf-8", errors="ignore")

            for line in stderr_output.splitlines():
                if "Stream" in line and "Video" in line:
                    size_match = re.search(r"(\d{2,})x(\d+)", line)
                    if size_match:
                        width, height = map(int, size_match.group(0).split("x"))
                    tbr_match = re.search(r"([\d\.]+) tbr", line)
                    if tbr_match:
                        fps = float(tbr_match.group(1))
                    else:
                        fps_match = re.search(r"([\d\.]+) fps", line)
                        if fps_match:
                            fps = float(fps_match.group(1))
                    break

            duration_match = re.search(r"Duration: (\d{2}):(\d{2}):(\d{2})\.(\d+)", stderr_output)
            if duration_match:
                h, m, s, ms_part = duration_match.groups()
                duration = (
                    int(h) * 3600
                    + int(m) * 60
                    + int(s)
                    + float(f"0.{ms_part}")
                )

        except FileNotFoundError as e:
            raise RuntimeError("未检测到 ffmpeg，可在系统 PATH 中安装。") from e

        return width, height, fps, duration

    def _extract_frames(self, filepath: str, width: int, height: int):
        if width <= 0 or height <= 0:
            raise RuntimeError("无法确定视频分辨率。")

        command = [
            "ffmpeg",
            "-i",
            filepath,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "pipe:1",
        ]

        frame_size = width * height * 4
        frames = []

        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10 ** 7,
        ) as proc:
            try:
                while True:
                    frame_bytes = proc.stdout.read(frame_size)
                    if not frame_bytes or len(frame_bytes) < frame_size:
                        break
                    frame_np = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((height, width, 4))
                    frames.append(frame_np.astype(np.float32) / 255.0)
            finally:
                proc.stdout.close()
                proc.wait()

        if not frames:
            raise RuntimeError("未能从视频提取任何帧。")

        return frames

    def load_video(self, url: str):
        if not url:
            raise ValueError("URL 输入为空。")

        temp_path = self._download_video(url)
        try:
            width, height, fps, source_duration = self._get_video_metadata(temp_path)
            if fps <= 0:
                fps = 30.0

            frames_np = self._extract_frames(temp_path, width, height)
            image_tensor = torch.from_numpy(np.stack(frames_np))

            loaded_frames = image_tensor.shape[0]
            loaded_duration = loaded_frames / fps if fps > 0 else 0.0
            source_duration = source_duration or loaded_duration
            source_frame_count = int(round(source_duration * fps)) if source_duration and fps > 0 else loaded_frames

            video_info = {
                "source_fps": fps,
                "source_frame_count": source_frame_count,
                "source_duration": source_duration,
                "source_width": width,
                "source_height": height,
                "loaded_fps": fps,
                "loaded_frame_count": loaded_frames,
                "loaded_duration": loaded_duration,
                "loaded_width": width,
                "loaded_height": height,
            }

            return (image_tensor, video_info)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class RegTuziChatResponse:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "response": ("STRING", {"forceInput": True}),
                "content_type": ("COMBO", {
                    "options": ["text", "image", "video"],
                    "default": "text",
                })
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("CONTENT",)
    FUNCTION = "reg_chat_response"
    CATEGORY = "BillBum_API/Utils"

    def reg_chat_response(self, response, content_type):
        
        if content_type == "text":
            out_str = response
        
        elif content_type == "image":
            all_urls = []
            image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tif', '.tiff']
            
            # 1. Use urlextract for robust http/https URL extraction
            try:
                extractor = URLExtract()
                http_urls = extractor.find_urls(response)
                for url in http_urls:
                    path = urlparse(url).path
                    if any(path.lower().endswith(ext) for ext in image_extensions):
                        all_urls.append(url)
            except Exception as e:
                print(f"urlextract failed for image: {e}. Falling back to regex.")
                # Fallback to regex if urlextract fails
                regex_http_urls = re.findall(
                    r'(https?://[^\s"\'<>)]+\.(?:' + '|'.join(ext.strip('.') for ext in image_extensions) + r')(?:\?[^\s"\'<>,)]*)?)',
                    response,
                    flags=re.IGNORECASE,
                )
                all_urls.extend(regex_http_urls)

            # 2. Always use regex for base64 data URIs
            base64_urls = re.findall(
                r'(data:image/[^;]+;base64,[^\s\)]+)',
                response,
                flags=re.IGNORECASE,
            )
            all_urls.extend(base64_urls)

            # 3. Deduplicate URLs while preserving order
            unique_urls = []
            seen = set()
            for url in all_urls:
                if url not in seen:
                    unique_urls.append(url)
                    seen.add(url)

            out_str = ",".join(unique_urls)

        elif content_type == "video":
            video_urls = []
            video_extensions = ['.mp4', '.webm', '.mov', '.mkv', '.avi', '.flv']
            
            # 1. Use urlextract for robust video URL extraction
            try:
                extractor = URLExtract()
                http_urls = extractor.find_urls(response)
                for url in http_urls:
                    path = urlparse(url).path
                    if any(path.lower().endswith(ext) for ext in video_extensions):
                        video_urls.append(url)
            except Exception as e:
                print(f"urlextract failed for video: {e}. Falling back to regex.")
                # Fallback to regex if urlextract fails
                regex_video_urls = re.findall(
                    r'(https?://[^\s"\'<>)]+\.(?:' + '|'.join(ext.strip('.') for ext in video_extensions) + r')(?:\?[^\s"\'<>,)]*)?)',
                    response,
                    flags=re.IGNORECASE
                )
                video_urls.extend(regex_video_urls)
            
            # 2. Deduplicate and get the last URL
            unique_urls = []
            seen = set()
            for url in video_urls:
                if url not in seen:
                    unique_urls.append(url)
                    seen.add(url)
            
            out_str = unique_urls[-1] if unique_urls else ""
            
        return (out_str,)
    
