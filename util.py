import tensorflow as tf
import numpy as np
import io
from pathlib import Path

# Specify the path to your model file
model_path = Path("model_vgg19.h5")

# Load the model
MODEL = tf.keras.models.load_model(model_path)
CLASSES = ["The patient is healthy", "Patient has Cancer"]

def load_img(img):
    # Reading The Image
    img = img.file.read()
    # Converting to Bytes Stream
    img = io.BytesIO(img)
    # Loading Image
    img = tf.keras.preprocessing.image.load_img(img, target_size=MODEL.input_shape[1:])
    # Converting Image to Array Of Correct Dimensions
    img = np.expand_dims(img, 0)
    # Returning Loaded Image
    return img

def preprocess(img):
    # Preprocessing the image
    img = img / 255.
    # Returning the processed image
    return img

def predict(X):
    pred = MODEL.predict(X)
    i = np.argmax(pred[0])
    return {
        "prediction": CLASSES[i],
        "accuracy": round(pred[0, i].tolist(), 3)
    }

def pipeline(img):
    loaded_img = load_img(img)
    processed_img = preprocess(loaded_img)
    response = predict(processed_img)

    return response
