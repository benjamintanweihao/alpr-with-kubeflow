## Setting up the project dependencies locally

### via Conda

```
conda create -n alpr -c conda-forge python=3.6 scipy==1.5.0  pandas==1.1.0 kfp==1.0.0 boto3==1.9.66 pip
conda activate alpr
pip install opencv-python==4.3.0.38 tensorflow-gpu==1.15.3 kfserving==0.4.0
```

Test it out:

```
python
Python 3.6.10 |Anaconda, Inc.| (default, May  8 2020, 02:54:21) 
[GCC 7.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow
>>> tensorflow.__version__
'1.15.0'
```

### Install Tensorflow Object Detection API

```
cd MODELS/research
# Compile protos.
protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
python -m pip install .

# Test the installation.
python object_detection/builders/model_builder_tf1_test.py
```
You should see all the tests pass:

```
Running tests under Python 3.6.10: /home/benjamintan/miniconda3/envs/alpr/bin/python
[ RUN      ] ModelBuilderTF1Test.test_create_context_rcnn_from_config_with_params(True)
[       OK ] ModelBuilderTF1Test.test_create_context_rcnn_from_config_with_params(True)
[ RUN      ] ModelBuilderTF1Test.test_create_context_rcnn_from_config_with_params(False)
[       OK ] ModelBuilderTF1Test.test_create_context_rcnn_from_config_with_params(False)
[ RUN      ] ModelBuilderTF1Test.test_create_experimental_model
[       OK ] ModelBuilderTF1Test.test_create_experimental_model
[ RUN      ] ModelBuilderTF1Test.test_create_faster_rcnn_from_config_with_crop_feature(True)
[       OK ] ModelBuilderTF1Test.test_create_faster_rcnn_from_config_with_crop_feature(True)
[ RUN      ] ModelBuilderTF1Test.test_create_faster_rcnn_from_config_with_crop_feature(False)
[       OK ] ModelBuilderTF1Test.test_create_faster_rcnn_from_config_with_crop_feature(False)
[ RUN      ] ModelBuilderTF1Test.test_create_faster_rcnn_model_from_config_with_example_miner
[       OK ] ModelBuilderTF1Test.test_create_faster_rcnn_model_from_config_with_example_miner
[ RUN      ] ModelBuilderTF1Test.test_create_faster_rcnn_models_from_config_faster_rcnn_with_matmul
[       OK ] ModelBuilderTF1Test.test_create_faster_rcnn_models_from_config_faster_rcnn_with_matmul
[ RUN      ] ModelBuilderTF1Test.test_create_faster_rcnn_models_from_config_faster_rcnn_without_matmul
[       OK ] ModelBuilderTF1Test.test_create_faster_rcnn_models_from_config_faster_rcnn_without_matmul
[ RUN      ] ModelBuilderTF1Test.test_create_faster_rcnn_models_from_config_mask_rcnn_with_matmul
[       OK ] ModelBuilderTF1Test.test_create_faster_rcnn_models_from_config_mask_rcnn_with_matmul
[ RUN      ] ModelBuilderTF1Test.test_create_faster_rcnn_models_from_config_mask_rcnn_without_matmul
[       OK ] ModelBuilderTF1Test.test_create_faster_rcnn_models_from_config_mask_rcnn_without_matmul
[ RUN      ] ModelBuilderTF1Test.test_create_rfcn_model_from_config
[       OK ] ModelBuilderTF1Test.test_create_rfcn_model_from_config
[ RUN      ] ModelBuilderTF1Test.test_create_ssd_fpn_model_from_config
[       OK ] ModelBuilderTF1Test.test_create_ssd_fpn_model_from_config
[ RUN      ] ModelBuilderTF1Test.test_create_ssd_models_from_config
[       OK ] ModelBuilderTF1Test.test_create_ssd_models_from_config
[ RUN      ] ModelBuilderTF1Test.test_invalid_faster_rcnn_batchnorm_update
[       OK ] ModelBuilderTF1Test.test_invalid_faster_rcnn_batchnorm_update
[ RUN      ] ModelBuilderTF1Test.test_invalid_first_stage_nms_iou_threshold
[       OK ] ModelBuilderTF1Test.test_invalid_first_stage_nms_iou_threshold
[ RUN      ] ModelBuilderTF1Test.test_invalid_model_config_proto
[       OK ] ModelBuilderTF1Test.test_invalid_model_config_proto
[ RUN      ] ModelBuilderTF1Test.test_invalid_second_stage_batch_size
[       OK ] ModelBuilderTF1Test.test_invalid_second_stage_batch_size
[ RUN      ] ModelBuilderTF1Test.test_session
[  SKIPPED ] ModelBuilderTF1Test.test_session
[ RUN      ] ModelBuilderTF1Test.test_unknown_faster_rcnn_feature_extractor
[       OK ] ModelBuilderTF1Test.test_unknown_faster_rcnn_feature_extractor
[ RUN      ] ModelBuilderTF1Test.test_unknown_meta_architecture
[       OK ] ModelBuilderTF1Test.test_unknown_meta_architecture
[ RUN      ] ModelBuilderTF1Test.test_unknown_ssd_feature_extractor
[       OK ] ModelBuilderTF1Test.test_unknown_ssd_feature_extractor
----------------------------------------------------------------------
Ran 21 tests in 0.144s

OK (skipped=1)
```

## Datasets

Create a `DATASETS` folder under the root directory:

```
mkdir DATASETS
wget -O DATASETS.tar.xz https://www.dropbox.com/s/qtowh6tq57kd2ss/DATASETS.tar.xz?dl=1'
tar xvf DATASETS.tar.xz
mkdir TFRECORDS
```

#### Create TFRecords


```
python dataprep/scripts/xml2csv.py -i DATASETS/romanian/train/annotations/ -o DATASETS/romanian/train/annotations/train_labels.csv
python dataprep/scripts/xml2csv.py -i DATASETS/romanian/test/annotations/ -o DATASETS/romanian/test/annotations/test_labels.csv

python dataprep/scripts/xml2csv.py -i DATASETS/indian/train/annotations/ -o DATASETS/indian/train/annotations/train_labels.csv
python dataprep/scripts/xml2csv.py -i DATASETS/indian/test/annotations/ -o DATASETS/indian/test/annotations/test_labels.csv

python dataprep/scripts/xml2csv.py -i DATASETS/voc/train/annotations/ -o DATASETS/voc/train/annotations/train_labels.csv
python dataprep/scripts/xml2csv.py -i DATASETS/voc/test/annotations/ -o DATASETS/voc/test/annotations/test_labels.csv

python dataprep/scripts/csv2tfrecords.py --split_name=train --tfrecord_name=romanian --label=license-plate --csv_input=DATASETS/romanian/train/annotations/train_labels.csv  --output_path=TFRECORDS --img_path=DATASETS/romanian/train/images/ 

python dataprep/scripts/csv2tfrecords.py --split_name=test --tfrecord_name=romanian --label=license-plate --csv_input=DATASETS/romanian/test/annotations/test_labels.csv  --output_path=TFRECORDS --img_path=DATASETS/romanian/test/images/ 

python dataprep/scripts/csv2tfrecords.py --split_name=train --tfrecord_name=indian --label=license-plate --csv_input=DATASETS/indian/train/annotations/train_labels.csv  --output_path=TFRECORDS --img_path=DATASETS/indian/train/images/ 

python dataprep/scripts/csv2tfrecords.py --split_name=test --tfrecord_name=indian --label=license-plate --csv_input=DATASETS/indian/test/annotations/test_labels.csv  --output_path=TFRECORDS --img_path=DATASETS/indian/test/images/ 

python dataprep/scripts/csv2tfrecords.py --split_name=train --tfrecord_name=voc --label=license-plate --csv_input=DATASETS/voc/train/annotations/train_labels.csv  --output_path=TFRECORDS --img_path=DATASETS/voc/train/images/ 

python dataprep/scripts/csv2tfrecords.py --split_name=test --tfrecord_name=voc --label=license-plate --csv_input=DATASETS/voc/test/annotations/test_labels.csv  --output_path=TFRECORDS --img_path=DATASETS/voc/test/images/ 
```

## Running Training

```
cd train
wget -O weights.tar.xz https://www.dropbox.com/s/bmdxebtj1cfk9ig/weights.tar.xz?dl=1
tar xvf weights.tar.xz
cd ..
```

Currently the only model available is SSD Inception V2, though you are free to download your own from the Model Zoo.

To configure training, make sure the following are set in `train/model_configs/ssd_inception_v2_coco.config`:


```
1.  Change num_classes to 1 as we have only one class of objects.
2.  fine_tune_checkpoint: “pre-trained-model/model.ckpt” to point to the pretrained model
3.  input_path for both train and test sets.
4.  label_map to point to our label map in the annotations directory
5.  num_steps is set to 20000 which is sufficient for our use case
```

```bash
python train/scripts/model_main.py --model_dir LOGS --pipeline_config_path train/model_configs/ssd_inception_v2_coco.config
```

### Observing via Tensorboard

```
$ tensorboard --logdir=LOGS
```

## Export Model

```bash
python MODELS/research/object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path train/model_configs/ssd_inception_v2_coco.config --trained_checkpoint_prefix LOGS/model.ckpt-20000 --output_directory SAVED_MODEL
``` 

Model is saved to `SAVED_MODEL/saved_model`

## Inference

```bash
python inference/scripts/detect.py --frozen_graph=SAVED_MODEL/frozen_inference_graph.pb --input_dir=DATASETS/indian/test/images/
```

You can find the results in the `inference/results` folder.

## Model Serving

Example:

```
docker run -t --rm -p 8500:8500 -p 8501:8501 -v \
"/home/benjamintan/workspace/servedmodels/ssd_inception_v2_coco_2018_01_28/:/models/ssdv2" -e \
MODEL_NAME=ssd_inception_v2_coco  tensorflow/serving:1.15.0
```

In `serving/rest_client.py`, set:

```
HOST = "localhost"
PORT = "8501"
MODEL_NAME = "ssd-inception-v2"
```

Execute the program. You should be able to observe the JSON output and resulting inference from `out.png`.
