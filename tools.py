from langchain.tools import BaseTool
from functions import get_image_caption
from functions import get_object_detection

class ImageCaptionTool(BaseTool):
    name = "Image caption tool"
    description = "Use this tool when given the path to an image that you would like to be described." \
                  "It will return a simple caption describing the image."

    def _run(self, image_path):
        return get_image_caption(image_path)

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async execution.")
    

class ObjectDetectionTool(BaseTool):
    name = "Object detection tool"
    description = "Use this tool when given the path to an image that you would like to detect objects." \
                  "It will return a list of detected objects with their labels and bounding boxes." \
                  "[x1, y1, x2, y2] class_name confidence_score"
    
    def _run(self, image_path):
        return get_object_detection(image_path) 
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async execution.")
    
