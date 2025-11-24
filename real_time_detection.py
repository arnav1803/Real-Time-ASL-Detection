import cv2
import numpy as np
import tensorflow as tf
import time

def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_frame(frame, target_size=(224, 224)):
    # Resize to model input size
    img = cv2.resize(frame, target_size)
    # Convert to RGB (OpenCV uses BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Preprocess input (MobileNetV2 expects [-1, 1])
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img

def main():
    model_path = 'asl_model.h5' # Or 'asl_model_final.h5'
    
    # Check if model exists
    import os
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found. Please train the model using 'asl_training.ipynb' first.")
        return

    model = load_model(model_path)
    if model is None:
        return

    # Classes A-Z
    classes = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Define a Region of Interest (ROI) for hand detection
        # This helps in reducing background noise. 
        # User should place hand in the box.
        height, width, _ = frame.shape
        roi_size = 300
        x1 = int(width / 2 - roi_size / 2)
        y1 = int(height / 2 - roi_size / 2)
        x2 = x1 + roi_size
        y2 = y1 + roi_size
        
        # Draw ROI on frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Extract ROI
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            continue

        # Preprocess ROI
        processed_roi = preprocess_frame(roi)

        # Predict
        start_time = time.time()
        predictions = model.predict(processed_roi, verbose=0)
        inference_time = (time.time() - start_time) * 1000 # ms
        
        predicted_class_idx = np.argmax(predictions)
        confidence = np.max(predictions)
        predicted_label = classes[predicted_class_idx]

        # Display result
        text = f"Pred: {predicted_label} ({confidence:.2f})"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"Inference: {inference_time:.1f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow('ASL Real-time Detection', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
