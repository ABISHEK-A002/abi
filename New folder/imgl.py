import cv2
import boto3

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

    # Step 2: Analyze the captured image using AWS Rekognition
    try:
        # Initialize AWS Rekognition client with specified region
        client = boto3.client('rekognition', region_name='us-west-2')  # Change to your desired region

        # Open the saved image file
        with open(image_path, 'rb') as image_file:
            response = client.detect_labels(Image={'Bytes': image_file.read()})

        # Extract and print the detected labels
        labels = [label['Name'] for label in response['Labels']]
        print(f"Detected objects: {', '.join(labels)}")

    except Exception as e:
        print(f"Error analyzing the image: {str(e)}")

# Run the function to capture and analyze the image
capture_and_analyze_image()
