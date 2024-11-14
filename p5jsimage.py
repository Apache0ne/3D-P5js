import base64
import nodes
import folder_paths
import os
import time
import torch
import torchvision.transforms as transforms
from aiohttp import web
from io import BytesIO
from PIL import Image
from server import PromptServer
import os
from aiohttp import web

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

    #OUTPUT_NODE = False

    CATEGORY = "p5js"

    def run(s, script, image, input_image=None):
        if input_image is not None:
            # Save input_image as "image.png" in the node's directory
            image_path = os.path.join(os.path.dirname(__file__), "image.png")
            torchvision.utils.save_image(input_image, image_path)
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
