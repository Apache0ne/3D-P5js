import base64
import nodes
import folder_paths
import os
import time
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from aiohttp import web
from io import BytesIO
import numpy as np
from PIL import Image
from server import PromptServer

# handle proxy response
@PromptServer.instance.routes.post('/HYPE/proxy_reply')
async def proxyHandle(request):
    post = await request.json()
    MessageHolder.addMessage(post["node_id"], post["outputs"])
    return web.json_response({"status": "ok"})

@PromptServer.instance.routes.get('/custom_node_image/{filename}')
async def serve_custom_node_image(request):
    filename = request.match_info['filename']
    file_path = os.path.join(os.path.dirname(__file__), 'web', filename)
    if os.path.exists(file_path):
        return web.FileResponse(file_path)
    return web.Response(status=404)

class HYPE_P5JSImage(nodes.LoadImage):
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "script": ("STRING", {"default": "function setup() {\n  createCanvas(512, 512);\n}\n\nfunction draw() {\n  background(220);\n}", "multiline": True, "dynamicPrompts": False}),
                "image": ("P5JS", {}),
            },
            "optional": {
                "input_image": ("IMAGE",),
            }
        }

    def IS_CHANGED(id):
        return True

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "run"

    CATEGORY = "p5js"

    def run(s, script, image, input_image=None):
        if input_image is not None:
            image_path = os.path.join(os.path.dirname(__file__), "web", "input_image.png")
            
            # Convert the input_image to a numpy array
            np_image = input_image.cpu().numpy()
            
            # Check if the image is a single pixel (1, 1, 3) with float values
            if np_image.shape == (1, 1, 3) and np_image.dtype == np.float32:
                # Expand the single pixel to a larger image
                np_image = np.repeat(np_image, 512, axis=0)
                np_image = np.repeat(np_image, 512, axis=1)
            
            # Ensure the image is in the correct format (H, W, C)
            if len(np_image.shape) == 4:
                np_image = np_image.squeeze(0)  # Remove batch dimension if present
            
            if len(np_image.shape) == 3 and np_image.shape[0] in [1, 3, 4]:
                np_image = np_image.transpose(1, 2, 0)
            
            # Normalize the image to 0-255 range and convert to uint8
            if np_image.dtype == np.float32:
                np_image = (np_image * 255).clip(0, 255).astype(np.uint8)
            
            # If the image is grayscale, convert to RGB
            if len(np_image.shape) == 2 or (len(np_image.shape) == 3 and np_image.shape[2] == 1):
                np_image = np.stack((np_image,) * 3, axis=-1)
            
            # Create a PIL Image and save it
            pil_image = Image.fromarray(np_image)
            pil_image.save(image_path)
        
        return super().load_image(folder_paths.get_annotated_filepath(image))

# Message Handling
class MessageHolder:
    messages = {}

    @classmethod
    def addMessage(self, id, message):
        self.messages[str(id)] = message

    @classmethod
    def waitForMessage(self, id, period = 0.1):
        sid = str(id)
        while not (sid in self.messages):
            time.sleep(period)
        message = self.messages.pop(str(id),None)
        return message