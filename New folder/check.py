import cv2
import numpy as np

def capture_and_analyze_image():
    # Step 1: Capture an image from the camera
    cap = cv2.VideoCapture(0)

    # Capture a frame from the camera
    ret, frame = cap.read()

    # Save the image locally if capture is successful
    if ret:
        image_path = 'captured_image.jpg'
        cv2.imwrite(image_path, frame)
        print(f"Image saved at {image_path}")
    else:
        print("Failed to capture image.")
        return

    cap.release()

    # Step 2: Load the pre-trained model
    net = cv2.dnn.readNetFromCaffe(
        'MobileNetSSD_deploy.prototxt', 
        'MobileNetSSD_deploy.caffemodel'
    )

    # Define the classes that the model can detect
    classes = ["background", "aeroplane", "bicycle", "bird", "boat", 
               "bottle", "bus", "car", "cat", "chair", "cow", 
               "diningtable", "dog", "horse", "motorbike", "person", 
               "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    # Step 3: Prepare the image for the model
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Step 4: Analyze the results
    detected_objects = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # Confidence score
        if confidence > 0.2:  # Set a confidence threshold
            class_id = int(detections[0, 0, i, 1])  # Class ID
            detected_objects.append(classes[class_id])  # Get the class name

    # Print the detected objects
    print(f"Detected objects: {', '.join(set(detected_objects))}")

# Run the function to capture and analyze the image
capture_and_analyze_image()
