import requests
import json
import cv2
import numpy as np

image = cv2.imread("/home/benjamintan/workspace/alpr-with-kubeflow/DATASETS/romanian/test/images/dayride_type1_001.mp4#t=31.jpg", 1)
image_content = image.astype('uint8').tolist()

instance = [{"inputs": image_content}]
data = json.dumps({"instances": instance, "signature_name": "serving_default"})

headers = {"content-type": "application/json"}

HOST = "10.1.1.242"
PORT = "8080"
MODEL_NAME = "old"

# To test on tensorflow/serving:
# docker run -t --rm -p 8500:8500 -p 8501:8501 -v \
# "/home/benjamintan/workspace/servedmodels/ssd_inception_v2_coco_2018_01_28/:/models/ssdv2" -e \
# MODEL_NAME=ssd_inception_v2_coco  tensorflow/serving:1.15.0
#
# HOST = "localhost"
# PORT = "8501"
# MODEL_NAME = "ssd-inception-v2"

json_response = requests.get(f"http://{HOST}:{PORT}/v1/models/{MODEL_NAME}/metadata")
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

    if score > 0.3:
        dim = image.shape
        box = box_normal_to_pixel(box, dim)
        b = box.astype(int)

        class_label = str(cls)

        # draw the image and write out
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 1)
        cv2.putText(image, class_label + "-" + str(round(score, 2)), (b[0] + 2, b[1] + 8), cv2.FONT_HERSHEY_SIMPLEX,
                    .45, (0, 0, 255))
        cv2.imwrite('out.png', image)
