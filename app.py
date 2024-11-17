import cv2
import numpy as np
import tensorflow as tf

# Load your trained model (assuming you're using a .h5 file)
model = tf.keras.models.load_model('webapp/models/traffic_dector_main.h5')

# Open the default camera (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()  # Read a frame from the camera feed

    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Pre-process the frame for the model (assuming your model uses 224x224 input size)
    input_frame = cv2.resize(frame, (224, 224))  # Resize to the input size of your model
    input_frame = input_frame / 255.0  # Normalize if the model was trained with normalized inputs

    # Add batch dimension (your model expects a batch of images)
    input_frame = np.expand_dims(input_frame, axis=0)

    # Run inference on the frame
    predictions = model.predict(input_frame)

    # Post-process the predictions (e.g., if you're detecting bounding boxes or classification)
    # Let's assume the model outputs bounding box coordinates and class probabilities

    # Example: Assuming the model outputs bounding boxes and classes (e.g., traffic objects)
    # predictions might look like this: [(x, y, w, h, class_id), ...]
    # In this example, we'll assume the model output is in the format of (bbox_x, bbox_y, bbox_width, bbox_height)
    
    # For demonstration, let's assume that the model predicts a list of bounding boxes (simplified)
    for prediction in predictions:
        # Assuming predictions contain bounding box coordinates (x, y, w, h)
        x, y, w, h = prediction[0], prediction[1], prediction[2], prediction[3]

        # Draw a bounding box on the frame (assuming it's a car or traffic object)
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

        # Optionally, add label for class if available
        label = "Car"  # Replace with class label if applicable
        cv2.putText(frame, label, (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with detections
    cv2.imshow('Traffic Detector', frame)

    # Exit condition: Press 'q' to quit the live camera feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
