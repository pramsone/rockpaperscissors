import streamlit as st
import tensorflow as tf
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
from google.colab import files
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

# Load the model
model = load_model("keras_model.h5")

# Load the labels
class_names = open("labels.txt", "r").readlines()

#Mencoba memprediksi melalui gambar yang ada di komputer
uploaded = files.upload()

for fn in uploaded.keys():
    # Load and preprocess the image
    path = fn
    img = image.load_img(path, target_size=(150, 150))
    imgplot = plt.imshow(img)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # Normalize the pixel values (assuming your model was trained with normalized data)

    # Predict the class
    classes = model.predict(x, batch_size=1)

    # Define your class labels
    class_labels = ["paper", "rock", "scissors"]

    print(fn)
    predicted_class = class_labels[np.argmax(classes)]
    print(f'Predicted class: {predicted_class}')