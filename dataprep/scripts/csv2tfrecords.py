"""
Usage:

# Create train data:
python csv2tfrecords.py --label=<LABEL> --csv_input=<PATH_TO_ANNOTATIONS_FOLDER>/train_labels.csv  --output_path=<PATH_TO_ANNOTATIONS_FOLDER>/train.record

# Create test data:
python csv2tfrecords.py --label=<LABEL> --csv_input=<PATH_TO_ANNOTATIONS_FOLDER>/test_labels.csv  --output_path=<PATH_TO_ANNOTATIONS_FOLDER>/test.record

eg:

python csv2tfrecords.py  --split_name=train --tfrecord_name=indian --label=license-plate --csv_input=../../DATASETS/indian/train/annotations/train_labels.csv --output_path ../../TFRECORDS/ --img_path=../../DATASETS/indian/train/images/
python csv2tfrecords.py  --split_name=test --tfrecord_name=indian --label=license-plate --csv_input=../../DATASETS/indian/test/annotations/test_labels.csv --output_path ../../TFRECORDS/ --img_path=../../DATASETS/indian/test/images/

python csv2tfrecords.py  --split_name=train --tfrecord_name=romanian --label=license-plate --csv_input=../../DATASETS/romanian/train/annotations/train_labels.csv --output_path ../../TFRECORDS/ --img_path=../../DATASETS/romanian/train/images/
python csv2tfrecords.py  --split_name=test --tfrecord_name=romanian --label=license-plate --csv_input=../../DATASETS/romanian/test/annotations/test_labels.csv --output_path ../../TFRECORDS/ --img_path=../../DATASETS/romanian/test/images/

python csv2tfrecords.py  --split_name=train --tfrecord_name=voc --label=license-plate --csv_input=../../DATASETS/voc/train/annotations/train_labels.csv --output_path ../../TFRECORDS/ --img_path=../../DATASETS/voc/train/images/
python csv2tfrecords.py  --split_name=test --tfrecord_name=voc --label=license-plate --csv_input=../../DATASETS/voc/test/annotations/test_labels.csv --output_path ../../TFRECORDS/ --img_path=../../DATASETS/voc/test/images/

"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import math
import os
import io
import pandas as pd
import tensorflow as tf
import sys

sys.path.append("../MODELS/research")

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label', 'license-plate', 'Name of class label')
flags.DEFINE_string('img_path', '', 'Path to images')
flags.DEFINE_string('split_name', '', 'train or test')
flags.DEFINE_string('tfrecord_name', '', 'name of TFRecord')
FLAGS = flags.FLAGS


def class_text_to_int(row_label):
    if row_label == FLAGS.label:
        return 1
    else:
        raise NotImplementedError


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    # check if the image format is matching with your images.
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def get_dataset_filename(dataset_dir, split_name, shard_id, tfrecord_filename, num_shards):
    output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (
        tfrecord_filename, split_name, shard_id, num_shards)
    return os.path.join(dataset_dir, output_filename)


def main(_):
    path = os.path.join(os.getcwd(), FLAGS.img_path)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')

    output_path = os.path.join(os.getcwd(), FLAGS.output_path)

    num_shards = 3
    split_name = FLAGS.split_name
    tfrecord_filename = FLAGS.tfrecord_name

    print(tfrecord_filename)

    num_per_shard = int(math.ceil(len(grouped) / float(num_shards)))

    with tf.Graph().as_default():
        with tf.Session() as sess:
            for shard_id in range(num_shards):
                output_filename = get_dataset_filename(output_path, split_name, shard_id, tfrecord_filename, num_shards)

                with tf.io.TFRecordWriter(output_filename) as writer:
                    start_idx = shard_id * num_per_shard
                    end_idx = min((shard_id + 1) * num_per_shard, len(grouped))
                    for i in range(start_idx, end_idx):
                        tf_example = create_tf_example(grouped[i], path)
                        writer.write(tf_example.SerializeToString())

    print('Successfully created the TFRecords: {}'.format(output_filename))


if __name__ == '__main__':
    tf.compat.v1.app.run()
