echo $@

# Set up the tensorflow object detection API
cd $1/MODELS/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf1/setup.py . && python -m pip install --user .
cd $1 && wget -O weights.tar.xz https://www.dropbox.com/s/bmdxebtj1cfk9ig/weights.tar.xz?dl=1 && tar xvf weights.tar.xz
export PYTHONPATH=$PYTHONPATH:$1/MODELS

cd $1 && python train/scripts/model_main.py --model_dir $5/LOGS/$2 \
                                            --pipeline_config_path train/model_configs/$2.config \
                                            --num_train_steps=$3 \
                                            --datasets_dir=$4/DATASETS \
                                            --tfrecords_dir=$4/TFRECORDS
