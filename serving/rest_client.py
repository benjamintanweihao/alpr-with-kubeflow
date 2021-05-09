import os

import requests
import json
import cv2
import numpy as np

from constants import MODEL_NAME

image_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), 'beach.jpeg'))
image = cv2.imread(image_path, 1)
image_content = image.astype('uint8').tolist()

instance = [{"inputs": image_content}]
data = json.dumps({"instances": instance, "signature_name": "serving_default"})

headers = {"content-type": "application/json"}

HOST = "POD-ID-GOES-HERE"
PORT = "8080"

THRESHOLD = 0.3

url = f"http://{HOST}:{PORT}/v1/models/{MODEL_NAME}/metadata"
print(url)

json_response = requests.get(url)
print(json_response.json())

json_response = requests.post(f"http://{HOST}:{PORT}/v1/models/{MODEL_NAME}:predict", data=data, headers=headers)
print(json_response.json())

predictions = json_response.json()['predictions'][0]
boxes = predictions['detection_boxes']
scores = predictions['detection_scores']
classes = predictions['detection_classes']


def box_normal_to_pixel(box, dim, scalefactor=1):
    height, width = dim[0], dim[1]
    ymin = int(box[0] * height * scalefactor)
    xmin = int(box[1] * width * scalefactor)

    ymax = int(box[2] * height * scalefactor)
    xmax = int(box[3] * width * scalefactor)
    return np.array([xmin, ymin, xmax, ymax])


for box, score, cls in zip(boxes, scores, classes):
    if score > THRESHOLD:
        dim = image.shape
        box = box_normal_to_pixel(box, dim)
        b = box.astype(int)

        class_label = str(cls)

        # draw the image and write out
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 1)
        cv2.putText(image, class_label + "-" + str(round(score, 2)), (b[0] + 2, b[1] + 8), cv2.FONT_HERSHEY_SIMPLEX,
                    .45, (0, 0, 255))
        cv2.imwrite('out.png', image)
