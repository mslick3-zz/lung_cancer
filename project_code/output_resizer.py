"""
inputs are positional args - input path, output path, output image size in form of depth,height,width
e.g. output_resizer.py ../../../output/processed_images_1  ../../../output/processed_images_1a 5,60,60
"""
from skimage.transform import resize
import argparse
import sys, os
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('input_path', action='store', help='input path')

    parser.add_argument('output_path', action='store', help='output path')

    parser.add_argument('shape', action='store', help='output shape')

    parser.add_argument('threshold', action='store', help='threshold', default=None)

    params = parser.parse_args()

    shape = np.array(params.shape.split(',')).astype(np.int)

    files = os.listdir(params.input_path)

    if not os.path.isdir(params.output_path):
        os.makedirs(params.output_path)

    for file in files:
        output_file = os.path.join(params.output_path, file)
        array = np.load(os.path.join(params.input_path, file))
        array = resize(array, shape, order=5, preserve_range=True)
        if params.threshold is not None:
            threshold = float(params.threshold)
            array_out = array[:]
            array_out[array<threshold] = 0.0
            array_out[array>=threshold] = 1.0
        np.save(output_file, array)