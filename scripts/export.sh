echo $@
echo $1
echo $2
echo $3
echo $4
echo $5
echo $6

cd $1/MODELS/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf1/setup.py . && python -m pip install --user .

python $1/MODELS/research/object_detection/export_inference_graph.py \
  --input_type image_tensor \
  --pipeline_config_path $1/train/model_configs/$2.config \
  --trained_checkpoint_prefix $4/LOGS/$2/model.ckpt-$3 \
  --output_directory $6/SAVED_MODEL/$2/$5

#  The model server expects that the saved model is in a versioned folder: e.g. sklearn-iris/4/saved_model.pb
mv $6/SAVED_MODEL/$2/$5/saved_model/saved_model.pb $6/SAVED_MODEL/$2/$5/saved_model.pb
rm -rf $6/SAVED_MODEL/$2/$5/saved_model/
