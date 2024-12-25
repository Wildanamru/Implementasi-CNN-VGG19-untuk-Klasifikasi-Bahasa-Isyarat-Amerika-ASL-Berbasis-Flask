import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, Response, request

# Initialize Flask app
app = Flask(__name__)

# Load the trained models
model_vgg = tf.keras.models.load_model('model2_asl_inceptionv3.h5')
model_resnet = tf.keras.models.load_model('model2_asl_resnet101.h5')
model_inception = tf.keras.models.load_model('model2_asl_inceptionv3.h5')  # New model

# Define class labels (A-Z, del, nothing, space)
class_labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Video capture setup
cap = cv2.VideoCapture(0)

# Variable to store selected model
selected_model = model_vgg

def preprocess_image(frame):
    """Preprocess the cropped hand image for model input."""
    # Resize the image to the model's expected input size (224x224)
    img_resized = cv2.resize(frame, (224, 224))

    # Normalize the image
    img_normalized = np.array(img_resized) / 255.0

    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)

    return img_batch

def detect_and_predict(frame):
    """Detect hands, crop the hand region, and predict with the model."""
    # Convert frame to RGB as MediaPipe uses RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and detect hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Get the bounding box around the hand
            x_min = min([lm.x for lm in landmarks.landmark])
            y_min = min([lm.y for lm in landmarks.landmark])
            x_max = max([lm.x for lm in landmarks.landmark])
            y_max = max([lm.y for lm in landmarks.landmark])

            # Convert normalized values to pixel values
            h, w, _ = frame.shape
            x_min, y_min = int(x_min * w), int(y_min * h)
            x_max, y_max = int(x_max * w), int(y_max * h)

            # Add padding to the bounding box
            padding = 20  # Adjust padding as needed
            x_min = max(x_min - padding, 0)
            y_min = max(y_min - padding, 0)
            x_max = min(x_max + padding, w)
            y_max = min(y_max + padding, h)

            # Crop the hand area from the frame
            hand_image = frame[y_min:y_max, x_min:x_max]

            # Preprocess the cropped hand image
            img_batch = preprocess_image(hand_image)

            # Make a prediction using the selected model
            predictions = selected_model.predict(img_batch)
            predicted_class_idx = np.argmax(predictions)
            predicted_class = class_labels[predicted_class_idx]

            # Draw the bounding box with more refined edges and color
            color = (0, 255, 0)  # Default color for bounding box (green)
            if predicted_class == 'nothing':
                color = (0, 0, 255)  # Red for 'nothing'
            elif predicted_class == 'space':
                color = (255, 255, 0)  # Blue for 'space'

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 3)  # Thicker bounding box
            cv2.putText(frame, f"Prediction: {predicted_class}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return frame

def gen_frames():
    """Capture video frames and yield them as a response."""
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame for hand detection and prediction
        frame = detect_and_predict(frame)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        # Convert the image to bytes
        frame = buffer.tobytes()

        # Yield the frame as a response for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    """Render the index HTML page."""
    global selected_model
    default_model = 'vgg'  # Default model name
    return render_template('index.html', selected_model=default_model)

@app.route('/video_feed')
def video_feed():
    """Serve the video feed to the browser."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/select_model', methods=['POST'])
def select_model():
    """Handle model selection from the dropdown."""
    global selected_model  
    model_name = request.form.get('model')
    if model_name == 'vgg':
        selected_model = model_vgg
    elif model_name == 'resnet':
        selected_model = model_resnet
    elif model_name == 'inception':  # New model option
        selected_model = model_inception
    return render_template('index.html', selected_model=model_name)

if __name__ == '__main__':
    app.run(debug=True)
