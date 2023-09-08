import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

# Load the pre-trained model
model = tf.keras.models.load_model('saved_model/my_model')

def load_and_preprocess_image(img_path):
    """
    Load the image, preprocess it for the model and expand its dimensions.
    """
    img = image.load_img(img_path, target_size=(200, 200))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.  # assuming images should be normalized to [0,1]

# Predict images in the 'murat' directory
images_dir = 'predict photos'
image_files = [os.path.join(images_dir, fname) for fname in os.listdir(images_dir)]

predictions = []

for img_file in image_files:
    processed_image = load_and_preprocess_image(img_file)
    prediction = model.predict(processed_image)
    predictions.append(prediction)

for i, pred in enumerate(predictions):
    print(f"Prediction for image {image_files[i]}: {pred}")

