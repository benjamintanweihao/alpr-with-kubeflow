## Setting up the project dependencies locally

```
pip install -r env/requirements.txt
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

## Install Tensorflow Object Detection API

```
git clone https://github.com/tensorflow/models.git MODELS
cd MODELS/research
# Compile protos.
protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
cp object_detection/packages/tf1/setup.py .
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
```

These are the datasets used:

### Romanian License Plate Dataset

git clone https://github.com/RobertLucian/license-plate-dataset DATASETS/romanian-license-plate-dataset


### Preparing the Dataset

#### Create `label_map.pbtxt`

```
item {
    id: 1
    name: 'license-plate'
}
```

#### Create TFRecords

```
git clone https://github.com/RobertLucian/license-plate-dataset.git romanian-license-plate-dataset
```

```
alpr-with-kubeflow/dataprep/scripts$ python xml2csv.py -i ../../DATASETS/romanian-license-plate-dataset/dataset/train/annots/ -o ../../DATASETS/romanian-license-plate-dataset/dataset/train/annots/train_labels.csv

alpr-with-kubeflow/dataprep/scripts$ python xml2csv.py -i ../../DATASETS/romanian-license-plate-dataset/dataset/valid/annots/ -o ../../DATASETS/romanian-license-plate-dataset/dataset/valid/annots/valid_labels.csv


python csv2tfrecords.py --label=license-plate --csv_input=../../DATASETS/romanian-license-plate-dataset/dataset/train/annots/train_labels.csv  --output_path=../../DATASETS/romanian-license-plate-dataset/dataset/train/annots/train.record --img_path=../../DATASETS/romanian-license-plate-dataset/dataset/train/images/

python csv2tfrecords.py --label=license-plate --csv_input=../../DATASETS/romanian-license-plate-dataset/dataset/valid/annots/valid_labels.csv  --output_path=../../DATASETS/romanian-license-plate-dataset/dataset/valid/annots/valid.record --img_path=../../DATASETS/romanian-license-plate-dataset/dataset/valid/images/
```


```
1.  Change num_classes to 1 as we have only one class of objects.
2.  fine_tune_checkpoint: “pre-trained-model/model.ckpt” to point to the pretrained model
3.  input_path for both train and test sets.
4.  label_map to point to our label map in the annotations directory
5.  num_steps is set to 20000 which is sufficient for our use case
```

## Running Training

```
cd alpr-with-kubeflow/train/scripts
```

Download the models from the Model Zoo. Each model in the tar.gz file contains the `pipeline.config` file which we modify for our needs.

Pick the model you want:

```bash
python model_main.py --model_dir ../../LOGS/ --pipeline_config_path ../model_configs/ssd_inception_v2_coco.config
```

```bash
python model_main.py --model_dir ../../LOGS/ --pipeline_config_path ../model_configs/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.config 
```

### Observing via Tensorboard

```
(alpr) benjamintan@cerberus:~/workspace/alpr-with-kubeflow/train$ tensorboard --logdir=logs
```

## Export Model

```bash
(alpr) ~/workspace/alpr-with-kubeflow$ python MODELS/research/object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path train/model_configs/ssd_inception_v2_coco.config --trained_checkpoint_prefix LOGS/model.ckpt-20000 --output_directory SAVED_MODEL
``` 

Model is saved to `SAVED_MODEL/saved_model`

## Inference

```
python inference/scripts/detect.py --frozen_graph=SAVED_MODEL/frozen_inference_graph.pb --input_dir=DATASETS/indian/train/images/
```