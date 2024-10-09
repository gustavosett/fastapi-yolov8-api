import cv2
import numpy as np
from ultralytics import YOLO

class ImageInferencer:
    def __init__(self):
        # initiate model
        self.model = YOLO("yolov8x")
    
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
