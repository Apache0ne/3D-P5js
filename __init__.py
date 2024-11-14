from .p5jsimage import HYPE_P5JSImage

NODE_CLASS_MAPPINGS = {
	"HYPE_P5JSImage": HYPE_P5JSImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
	"HYPE_P5JSImage": "p5js image"
}

WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', "WEB_DIRECTORY"]
