import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("image_detector_model.keras")

# Settings (must match training)
img_height = 224
img_width = 224
class_names = ['FAKE', 'REAL']  # alphabetical order, same as training

def predict_image(img_path):
    # Load and preprocess the image
    img = tf.keras.utils.load_img(img_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0  # normalize, same as training
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    # Predict
    prediction = model.predict(img_array)[0][0]
    
    if prediction > 0.5:
        label = class_names[1]  # REAL
        confidence = prediction
    else:
        label = class_names[0]  # FAKE
        confidence = 1 - prediction

    print(f"Prediction: {label} (confidence: {confidence*100:.2f}%)")

# ---- Try it on an image ----
predict_image("dataset/test/REAL/0000.jpg")  # change this path to any image you want to test