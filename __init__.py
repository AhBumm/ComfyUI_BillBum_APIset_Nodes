from .billbum_modified import *
from .nodes4tuzi import (
    BillBum_Modified_StreamResponse_LLM_API,
    Url2Image,
    RegTuziChatResponse,
    LoadVideoFromUrlVHS,
    LoadVideoFromUrlComfyIO,
    )
from .nodes4hypr import HyprLab_Image_API_Node
from .nodes4doubao import(
    seedance_api_node,
    seedream_api_node,
    seedance2_api_node
    )
from .responses_api import BillBum_Modified_Responses_API_Node

# Exporting the node classes for ComfyUI to discover
NODE_CLASS_MAPPINGS = {
    "BillBum_Modified_Dalle_API_Node": BillBum_Modified_Dalle_API_Node,
    "BillBum_Modified_LLM_API_Node": BillBum_Modified_LLM_API_Node,
    "BillBum_Modified_img2b64_url_Node": BillBum_Modified_img2b64url_Node,
    "BillBum_Modified_VisionLM_API_Node": BillBum_Modified_VisionLM_API_Node,
    "BillBum_Modified_SD3_API_Node": BillBum_Modified_SD3_API_Node,
    "BillBum_Modified_Base64_Url2Img_Node": BillBum_Modified_Base64_Url2Img_Node,
    "BillBum_Modified_RegText_Node": BillBum_Modified_RegText_Node,
    "BillBum_Modified_DropoutToken_Node": BillBum_Modified_DropoutToken_Node,
    "BillBum_Modified_Image_API_Call_Node": BillBum_Modified_Image_API_Call_Node,
    "BillBum_Modified_Recraft_API_Node": BillBum_Modified_Recraft_API_Node,
    "Text_Concat": Text_Concat,
    "Input_Text": Input_Text,
    "BillBum_Modified_Ideogram_API_Node": BillBum_Modified_Ideogram_API_Node,
    "BillBum_NonSysPrompt_VLM_API_Node": BillBum_NonSysPrompt_VLM_API_Node,
    "BillBum_Modified_LLM_ForceStream_Mode": BillBum_Modified_LLM_ForceStream_Mode,
    "BillBum_Modified_GPTImage1_API_Node": BillBum_Modified_GPTImage1_API_Node,
    "BillBum_Modified_Flux_API_with_imgInput": BillBum_Modified_Flux_API_Node_imgInput,
    "BillBum_Modified_Responses_API_Node": BillBum_Modified_Responses_API_Node,
    "billbum_modified_stream_response_llm_api": BillBum_Modified_StreamResponse_LLM_API,
    "url2image": Url2Image,
    "reg_tuzi_chat_response": RegTuziChatResponse,
    "load_video_from_url": LoadVideoFromUrlVHS,
    "load_video_from_url_comfy_core": LoadVideoFromUrlComfyIO,
    "hyprlab_image_api_node": HyprLab_Image_API_Node,
    "doubao_seedance_api_node": seedance_api_node,
    "doubao_seedream_api_node": seedream_api_node,
    "doubao_seedance2_api_node": seedance2_api_node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "billbum_modified_stream_response_llm_api": "API Node for Stream Response LLMs",
    "url2image": "Load Image from URL (BillBum)",
    "reg_tuzi_chat_response": "UrlExtract from Chat Response",
    "load_video_from_url": "Load Video From URL (VHS Compatible)",
    "load_video_from_url_comfy_core": "Load&Save Video From URL (Comfy Core)",
    "BillBum_Modified_Dalle_API_Node": "Dall-E Custom API Node",
    "BillBum_Modified_LLM_API_Node": "Custom LLM API Node (Old)",
    "BillBum_Modified_img2b64_url_Node": "Image to Base64 URL Node",
    "BillBum_Modified_VisionLM_API_Node": "Vision LLMs API Node (Old)",
    "BillBum_Modified_SD3_API_Node": "Stable Diffusion 3 API Node",
    "BillBum_Modified_Base64_Url2Img_Node": "Base64 URL to Image Node",
    "BillBum_Modified_RegText_Node": "Regular ResponseText to 1linePrompt Node",
    "BillBum_Modified_DropoutToken_Node": "Dropout by MaxToken Node",
    "BillBum_Modified_Image_API_Call_Node": "Custom Image Generation API Call Node",
    "BillBum_Modified_Recraft_API_Node": "Custom Recraft API Node",
    "Text_Concat": "Concat Text Strings Node",
    "Input_Text": "Input Text",
    "BillBum_Modified_Ideogram_API_Node": "Custom Ideogram API Node",
    "BillBum_NonSysPrompt_VLM_API_Node": "Non-System Prompt VLMs API Node",
    "BillBum_Modified_LLM_ForceStream_Mode": "LLM StreamResponse Node (Old)",
    "BillBum_Modified_GPTImage1_API_Node": "Custom GPTImage1 API Node",
    "BillBum_Modified_Flux_API_with_imgInput": "Custom Flux API Node",
    "BillBum_Modified_Responses_API_Node": "OpenAI Responses API Node",
    "hyprlab_image_api_node": "HyprLab ImageGen API Node",
    "doubao_seedance_api_node": "Doubao Seedance VideoGen API Node",
    "doubao_seedream_api_node": "Doubao Seedream ImageGen API Node",
    "doubao_seedance2_api_node": "Doubao Seedance2 VideoGen API Node",
}