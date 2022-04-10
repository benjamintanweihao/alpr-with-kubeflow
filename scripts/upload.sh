#1 project_workspace
#2 model_dir
#3 model_name
#4 export_bucket

echo $@
python $1/export/upload_to_s3.py \
  --model_dir=$2 \
  --model_name=$3 \
  --export_bucket=$4
