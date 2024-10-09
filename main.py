import io
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from predictor import ImageInferencer

# start engine
app = FastAPI()
model = ImageInferencer()

@app.post("/inference-image")
async def inference_uploaded_image(file: UploadFile = File(...)):
    # read the uploaded file into bytes
    image_bytes = await file.read()
    
    # inference image
    output_image_bytes = model.inference_image(image_bytes)

    return StreamingResponse(io.BytesIO(output_image_bytes), media_type="image/jpeg")
