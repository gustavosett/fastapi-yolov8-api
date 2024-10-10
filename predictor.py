import cv2
import numpy as np
from ultralytics import YOLO

class Singleton(object):
  _instances = {}
  def __new__(class_, *args, **kwargs):
    if class_ not in class_._instances:
        class_._instances[class_] = super(Singleton, class_).__new__(class_, *args, **kwargs)
    return class_._instances[class_]

class ImageInferencer(Singleton):
    def __init__(self, model_version="yolov8x"):
        # initiate model
        self.model = YOLO(model_version)
    
    def inference_image(self, image_bytes) -> bytes | None:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # inference image
        results = self.model(img_np)
        
        # get the first result
        result = results[0]
        
        # plot results
        im_array = result.plot()
        
        # encode the annotated image back to bytes
        success, encoded_image = cv2.imencode('.jpg', im_array)
        if not success:
            return None
        return encoded_image.tobytes()
