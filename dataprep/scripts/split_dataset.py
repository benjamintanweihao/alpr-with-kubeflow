# Split dataset into test and train
import argparse
import shutil
import os
from random import randint

from sklearn.model_selection import train_test_split
from os import listdir
from os.path import isfile, join


def split_dataset(input_dir):
    images_dir = join(input_dir, 'images')
    annotations_dir = join(input_dir, 'annotations')

    images_train_dir = join(input_dir, 'train', 'images')
    annotations_train_dir = join(input_dir, 'train', 'annotations')

    images_test_dir = join(input_dir, 'test', 'images')
    annotations_test_dir = join(input_dir, 'test', 'annotations')

    images = sorted([join(images_dir, f) for f in listdir(images_dir) if isfile(join(images_dir, f))])
    annotations = sorted(
        [join(annotations_dir, f) for f in listdir(annotations_dir) if isfile(join(annotations_dir, f))])

    # Double check that image and annotation filenames match up
    for x, y in zip(images, annotations):
        name_x, _ = os.path.splitext(os.path.basename(x))
        name_y, _ = os.path.splitext(os.path.basename(y))
        assert name_x == name_y, f'{name_x} != {name_y}'

    X_train, X_test, y_train, y_test = train_test_split(images, annotations,
                                                        test_size=0.2,
                                                        shuffle=True,
                                                        random_state=randint(0, 2 ** 32 - 1))

    dirs_to_make = [images_train_dir, images_test_dir, annotations_train_dir, annotations_test_dir]
    [os.makedirs(d, exist_ok=False) for d in dirs_to_make]

    # Write train and test
    for x_src, y_src in zip(X_train, y_train):
        x_dst = join(images_train_dir, os.path.basename(x_src))
        y_dst = join(annotations_train_dir, os.path.basename(y_src))
        shutil.copy(x_src, x_dst)
        shutil.copy(y_src, y_dst)

    for x_src, y_src in zip(X_test, y_test):
        x_dst = join(images_test_dir, os.path.basename(x_src))
        y_dst = join(annotations_test_dir, os.path.basename(y_src))
        shutil.copy(x_src, x_dst)
        shutil.copy(y_src, y_dst)


def main():
    parser = argparse.ArgumentParser(description="XML-to-CSV converter")
    parser.add_argument("-i",
                        "--inputDir",
                        help="Path to the folder where the input dataset is stored",
                        type=str)
    args = parser.parse_args()
    split_dataset(args.inputDir)


if __name__ == '__main__':
    main()
