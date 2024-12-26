from .billbum_modified import *
# Exporting the node classes for ComfyUI to discover
NODE_CLASS_MAPPINGS = {
    "BillBum_Modified_Dalle_API_Node": BillBum_Modified_Dalle_API_Node,
    "BillBum_Modified_LLM_API_Node": BillBum_Modified_LLM_API_Node,
    "BillBum_Modified_img2b64_url_Node": BillBum_Modified_img2url_Node,
    "BillBum_Modified_VisionLM_API_Node": BillBum_Modified_VisionLM_API_Node,
    "BillBum_Modified_SD3_API_Node": BillBum_Modified_SD3_API_Node,
    "BillBum_Modified_Together_API_Node": BillBum_Modified_Together_API_Node,
    "BillBum_Modified_Base64_Url2Img_Node": BillBum_Modified_Base64_Url2Img_Node,
    "BillBum_Modified_ImageSplit_Node": BillBum_Modified_ImageSplit_Node,
    "BillBum_Modified_Base64_Url2Data_Node": BillBum_Modified_Base64_Url2Data_Node,
    "BillBum_Modified_Structured_LLM_Node(Imperfect)": BillBum_Modified_Structured_LLM_Node,
    "BillBum_Modified_Flux_API_Node": BillBum_Modified_Flux_API_Node,
    "BillBum_Modified_RegText_Node": BillBum_Modified_RegText_Node,
    "BillBum_Modified_DropoutToken_Node": BillBum_Modified_DropoutToken_Node,
    "BillBum_Modified_Image_API_Call_Node": BillBum_Modified_Image_API_Call_Node,
    "BillBum_Modified_Recraft_API_Node": BillBum_Modified_Recraft_API_Node,
    "Text_Concat": Text_Concat,
    "Input_Text": Input_Text,
    "BillBum_Modified_Ideogram_API_Node": BillBum_Modified_Ideogram_API_Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
}