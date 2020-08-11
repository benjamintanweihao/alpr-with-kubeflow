import numpy as np
import tensorflow as tf
import cv2 as cv
import os
from os import listdir

# Change PATH names to match your file paths for the model and INPUT_FILE'
PATH_TO_FROZEN_GRAPH = os.path.join('..', '..', 'train', 'frozen_inference_graph.pb')
INPUT_FILE = os.path.join('..', 'test_images', 'multiple_plates.png')

assert os.path.exists(PATH_TO_FROZEN_GRAPH)
assert os.path.exists(INPUT_FILE)

# Read the model from the file
with tf.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

# Create the tensorflow session
with tf.Session() as sess:
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    # Read the input file
    img = cv.imread(INPUT_FILE)

    rows = img.shape[0]
    cols = img.shape[1]
    inp = cv.resize(img, (300, 300))
    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

    # Run the model
    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                   feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

    # Visualize detected bounding boxes.
    num_detections = int(out[0][0])

    print(f'Num detections: {num_detections}')
    # Iterate through all detected detections
    for i in range(num_detections):
        classId = int(out[3][0][i])

        score = float(out[1][0][i])
        bbox = [float(v) for v in out[2][0][i]]

        print(f'Score: {score}')

        if score > 0.1:
            # Creating a box around the detected number plate
            x = int(bbox[1] * cols)
            y = int(bbox[0] * rows)
            right = int(bbox[3] * cols)
            bottom = int(bbox[2] * rows)
            cv.rectangle(img, (x, y), (right, bottom), (125, 255, 51), thickness=2)
            cv.imwrite('licence_plate_detected.png', img)
            cv.imshow('license_plate_detected', img )
            cv.waitKey(0)
            cv.destroyAllWindows()
