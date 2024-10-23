## Introduction 

BillBum Modified Comfyui Nodes is a set of nodes for myself to use api in comfyui.
including DALL-E, OpenAI's LLMs, other LLMs api platform, also other image generation api.
![screenshot](https://github.com/user-attachments/assets/3dd235f3-cfd1-45dd-8914-c2b0fd68e5f1)

## Features

- **Text Generation**: Use API to ask llm text generation, structured responses (not work yep).
- **Image Generation**: Use API to Generate images, support dalle and flux1.1-pro etc.
- **Vision LM**: Use API to caption or describe image, need models vision supported.
- **little tools**: base64 url to base64 data, base64 url to IMAGE, IMAGE to base64 url, regular llm text to word and "," only. etc.

## Installation
In ComfyUI Manager Menu "Install via Git URL"
```
https://github.com/AhBumm/ComfyUI_BillBum_Nodes.git
```
Or search "billbum" in ComfyUI Manager
![image](https://github.com/user-attachments/assets/86ec81bf-2fff-4875-9ce9-f122feac79d7)

install requirements with comfyui embeded python
```
pip install -r requirements.txt
```

## Update
- **Add use_jailbreak option for VisionLM api node**
  If your caption task rejected due to nsfw content, you can try to take on use_jailbreak.
  tested models:
  - llama-3.2-11b
  - llama-3.2-90b
  - gemini-1.5-flash
  - gemini-1.5-pro
  - pixtral-12b-latest
