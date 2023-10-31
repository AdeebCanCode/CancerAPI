from fastapi import FastAPI, File, UploadFile
from keras.models import load_model
from keras.applications.vgg19 import preprocess_input
import numpy as np
from PIL import Image
import io
from pathlib import Path

# Specify the path to your model file
model_path = Path("model_vgg19.h5")

# Load the pre-trained model
model = load_model(model_path)

# Create a FastAPI app
app = FastAPI()

@app.post("/predict/")
async def predict_image(file: UploadFile):
    try:
        # Check if the file is an image
        if file.content_type.startswith('image'):
            # Read and preprocess the image
            img = Image.open(io.BytesIO(await file.read()))
            img = img.resize((224, 224))
            x = np.array(img)
            x = np.expand_dims(x, axis=0)
            img_data = preprocess_input(x)

            # Make predictions
            classes = model.predict(img_data)
            malignant = classes[0, 0]
            normal = classes[0, 1]

            # Determine the result
            if malignant > normal:
                prediction = 'malignant'
            else:
                prediction = 'normal'

            return {"prediction": prediction}
        else:
            return {"error": "Invalid file format, please provide an image."}
    except Exception as e:
        return {"error": "Internal server error"}

if __name__ == "__main__":
    import os
    import uvicorn

    # Dynamically configure the port using the PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
