from .billbum_modified import *
# Exporting the node classes for ComfyUI to discover
NODE_CLASS_MAPPINGS = {
    "BillBum_Modified_Dalle_API_Node": BillBum_Modified_Dalle_API_Node,
    "BillBum_Modified_LLM_API_Node": BillBum_Modified_LLM_API_Node,
    "BillBum_Modified_LLM_API_sequentialNode": BillBum_Modified_LLM_API_sequentialNode,
    "BillBum_Modified_img2url_Node": BillBum_Modified_img2url_Node,
    "BillBum_Modified_VisionLM_API_Node": BillBum_Modified_VisionLM_API_Node,
    "BillBum_Modified_Ideogram_API_Node": BillBum_Modified_Ideogram_API_Node,
    "BillBum_Modified_Together_API_Node": BillBum_Modified_Together_API_Node,
    "BillBum_Modified_Base64_Url2Img_Node": BillBum_Modified_Base64_Url2Img_Node,
    "BillBum_Modified_ImageSplit_Node": BillBum_Modified_ImageSplit_Node,
    "BillBum_Modified_Base64_Url2Data_Node": BillBum_Modified_Base64_Url2Data_Node,
    "BillBum_Modified_Structured_LLM_Node(Imperfect)": BillBum_Modified_Structured_LLM_Node,
    "BillBum_Modified_T2I_API_Node": BillBum_Modified_Text2Image_API_Node,
    "BillBum_Modified_RegText_Node": BillBum_Modified_RegText_Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
}