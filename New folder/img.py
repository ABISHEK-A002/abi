import boto3

def analyze_image(image_path):
    client = boto3.client('rekognition')
    with open(image_path, 'rb') as image_file:
        response = client.detect_labels(Image={'Bytes': image_file.read()})
    labels = [label['Name'] for label in response['Labels']]
    description = ', '.join(labels)
    print(f"Detected objects: {description}")
    return description
