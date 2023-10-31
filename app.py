from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg19 import preprocess_input
import numpy as np
from PIL import Image
import io

# Load the pre-trained model using TensorFlow
model = tf.keras.models.load_model("model_vgg19_saved_model")

# Create a FastAPI app
app = FastAPI()

@app.post("/predict/")
async def predict_image(file: UploadFile):
    try:
        # Check if the file is an image
        if file.content_type.startswith("image"):
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
                prediction = "malignant"
            else:
                prediction = "normal"

            return {"prediction": prediction}
        else:
            return {"error": "Invalid file format, please provide an image."}
    except Exception as e:
        return {"error": "Internal server error"}

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
