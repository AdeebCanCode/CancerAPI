from fastapi import FastAPI, UploadFile, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
import numpy as np
import os

app = FastAPI()

# Define the full path to the model file
model_path = os.path.join(os.getcwd(), 'model_vgg19.h5')

# Load your Keras model using a FastAPI dependency
def load_model_dependency():
    return load_model(model_path)

def predict(image_path, model):
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        img_data = preprocess_input(x)
        classes = model.predict(img_data)
        malignant = float(classes[0, 0])  # Convert to float
        normal = float(classes[0, 1])     # Convert to float
        return malignant, normal
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error processing image")

@app.post("/predict/")
async def predict_image(file: UploadFile, model: 'Model' = Depends(load_model_dependency)):
    try:
        # Save the uploaded image temporarily
        with open("temp_image.jpg", "wb") as temp_image:
            temp_image.write(file.file.read())
        
        # Perform prediction on the saved image
        malignant, normal = predict("temp_image.jpg", model)
        
        # Clean up the temporary image file
        os.remove("temp_image.jpg")
        
        if malignant > normal:
            prediction = 'malignant'
        else:
            prediction = 'normal'
        
        # Convert NumPy floats to Python floats
        malignant = float(malignant)
        normal = float(normal)
        
        return {"prediction": prediction, "malignant_prob": malignant, "normal_prob": normal}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
