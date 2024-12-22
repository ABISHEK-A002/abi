from flask import Flask, jsonify
import cv2
import boto3

app = Flask(__name__)

def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        image_path = 'captured_image.jpg'
        cv2.imwrite(image_path, frame)
    cap.release()
    return image_path

def analyze_image(image_path):
    client = boto3.client('rekognition')
    with open(image_path, 'rb') as image_file:
        response = client.detect_labels(Image={'Bytes': image_file.read()})
    labels = [label['Name'] for label in response['Labels']]
    return ', '.join(labels)

@app.route('/what_do_you_see', methods=['GET'])
def what_do_you_see():
    image_path = capture_image()
    description = analyze_image(image_path)
    return jsonify({"description": description})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
