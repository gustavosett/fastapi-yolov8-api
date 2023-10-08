from ultralytics import YOLO
from PIL import Image

class ImageInferencer:
    def __init__(self):
        # initiate model
        self.model = YOLO("yolov8x")
    
    def __draw_detection(self, results, path):
        for result in results:
            # plot results
            im_array = result.plot()
            # turns BGR to RGB
            im = Image.fromarray(im_array[..., ::-1])
            # save
            im.save(path)
        
    def inference_image(self, path, new_path):
        # inference image
        results = self.model(path)
        
        # plot frame
        plotted_frame = self.__draw_detection(results, new_path)
        
        return plotted_frame