import tensorflow as tf
import numpy as np
from PIL import Image

# Load the saved model
model = tf.keras.models.load_model("model.keras")

# Function to preprocess the image
def preprocess_image(image_path):
    # Load the image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28 pixels
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Path to your image
image_path = r"C:\Users\Rapid IT World\Desktop\test sample\img_11.jpg"

# Preprocess the image
input_data = preprocess_image(image_path)

# Make predictions
predictions = model.predict(input_data)
predicted_label = np.argmax(predictions)

print(f"Predicted Label: {predicted_label}")
