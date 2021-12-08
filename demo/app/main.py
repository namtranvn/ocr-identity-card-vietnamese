import json
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"

    image = read_image_file(await file.read())
    prediction = predict(image)

    return prediction


if __name__ == "__main__":
    uvicorn.run(app, debug=True)