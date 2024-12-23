import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load the saved model
model = tf.keras.models.load_model("model.keras")

# Function to preprocess the image
def preprocess_image(image):
    # Convert to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize to 28x28 pixels
    img_resized = cv2.resize(gray_img, (28, 28))
    # Normalize pixel values to [0, 1]
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension for grayscale
    return img_array

# Open the default camera
vid = cv2.VideoCapture(0)

# Set the frame width and height
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Increased resolution
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    # Read a frame from the camera
    ret, frame = vid.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break
    
    # Define the region of interest (ROI) for digit capture
    roi = frame[100:300, 100:300]  # Adjust based on your frame
    input_data = preprocess_image(roi)
    
    # Make predictions
    predictions = model.predict(input_data)
    predicted_label = np.argmax(predictions)
    
    # Display the predicted label on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Predicted: {predicted_label}", (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Display the frame with the prediction
    cv2.imshow('frame', frame)
    
    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
vid.release()
cv2.destroyAllWindows()
