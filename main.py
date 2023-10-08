from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from pathlib import Path
from predictor import ImageInferencer
import shutil

# start engine
app = FastAPI()
model = ImageInferencer()

@app.post("/inference-image")
async def inference_uploaded_image(file: UploadFile = File()):
    # save paths
    path = f"{Path.cwd()}/images/{file.filename}"
    final_path = f"{Path.cwd()}/images/inferenced_{file.filename}"
    
    # save file
    with open(path, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
        
    # inference image
    model.inference_image(path, final_path)
    
    return FileResponse(final_path)

        
        
    