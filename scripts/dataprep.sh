echo $@

# Download dataset
mkdir -p $2/DATASETS

#cd $2 && wget -O DATASETS.tar.xz https://www.dropbox.com/s/qtowh6tq57kd2ss/DATASETS.tar.xz?dl=1 && tar xvf DATASETS.tar.xz
cd $2 && wget -O DATASETS.tar.xz https://www.dropbox.com/s/xv0e4ghdka9e1z5/DATASETS-tiny.tar.xz?dl=1 && tar xvf DATASETS.tar.xz -C $2/DATASETS --strip-components=1

mkdir -p $2/TFRECORDS

# Set up the tensorflow object detection API
cd $1/MODELS/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf1/setup.py . && python -m pip install --user .

python $1/dataprep/scripts/xml2csv.py -i $2/DATASETS/romanian/train/annotations/ -o $2/DATASETS/romanian/train/annotations/train_labels.csv
python $1/dataprep/scripts/xml2csv.py -i $2/DATASETS/romanian/test/annotations/ -o $2/DATASETS/romanian/test/annotations/test_labels.csv
python $1/dataprep/scripts/xml2csv.py -i $2/DATASETS/indian/train/annotations/ -o $2/DATASETS/indian/train/annotations/train_labels.csv
python $1/dataprep/scripts/xml2csv.py -i $2/DATASETS/indian/test/annotations/ -o $2/DATASETS/indian/test/annotations/test_labels.csv
python $1/dataprep/scripts/xml2csv.py -i $2/DATASETS/voc/train/annotations/ -o $2/DATASETS/voc/train/annotations/train_labels.csv
python $1/dataprep/scripts/xml2csv.py -i $2/DATASETS/voc/test/annotations/ -o $2/DATASETS/voc/test/annotations/test_labels.csv
python $1/dataprep/scripts/csv2tfrecords.py --split_name=train --tfrecord_name=romanian --label=license-plate --csv_input=$2/DATASETS/romanian/train/annotations/train_labels.csv  --output_path=$2/TFRECORDS --img_path=$2/DATASETS/romanian/train/images/
python $1/dataprep/scripts/csv2tfrecords.py --split_name=test --tfrecord_name=romanian --label=license-plate --csv_input=$2/DATASETS/romanian/test/annotations/test_labels.csv  --output_path=$2/TFRECORDS --img_path=$2/DATASETS/romanian/test/images/
python $1/dataprep/scripts/csv2tfrecords.py --split_name=train --tfrecord_name=indian --label=license-plate --csv_input=$2/DATASETS/indian/train/annotations/train_labels.csv  --output_path=$2/TFRECORDS --img_path=$2/DATASETS/indian/train/images/
python $1/dataprep/scripts/csv2tfrecords.py --split_name=test --tfrecord_name=indian --label=license-plate --csv_input=$2/DATASETS/indian/test/annotations/test_labels.csv  --output_path=$2/TFRECORDS --img_path=$2/DATASETS/indian/test/images/
python $1/dataprep/scripts/csv2tfrecords.py --split_name=train --tfrecord_name=voc --label=license-plate --csv_input=$2/DATASETS/voc/train/annotations/train_labels.csv  --output_path=$2/TFRECORDS --img_path=$2/DATASETS/voc/train/images/
python $1/dataprep/scripts/csv2tfrecords.py --split_name=test --tfrecord_name=voc --label=license-plate --csv_input=$2/DATASETS/voc/test/annotations/test_labels.csv  --output_path=$2/TFRECORDS --img_path=$2/DATASETS/voc/test/images/
