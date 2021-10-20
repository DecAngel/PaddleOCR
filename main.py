import os
import json
import functools
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

root = Path().joinpath('datasets', 'iron')


def preprocess():
    for category in ['train', 'test']:
        simple_pic_path = root.joinpath(category + '_simple')
        simple_pic_path.mkdir(exist_ok=True)
        pic_path = root.joinpath(category)
        with open(root.joinpath(category + '_label.txt'), 'r') as f:
            with open(root.joinpath(category + '_simple_label.txt'), 'w') as g:
                while True:
                    t = f.readline()
                    if t == '':
                        break
                    file_name, label = t.split('\t')
                    new_file_name = '{}_{}.{}'.format(*file_name.split('.'))
                    label_json = json.loads(label)
                    text, points = label_json[0]['transcription'], label_json[0]['points']
                    rect = functools.reduce(lambda x, y: (min(x[0], y[0]), min(x[1], y[1])), points) + functools.reduce(
                        lambda x, y: (max(x[0], y[0]), max(x[1], y[1])), points)
                    img: Image.Image = Image.open(str(pic_path.joinpath(file_name).resolve()))
                    rect = max(0, rect[0]-20), max(0, rect[1]-20), min(img.width, rect[2]+20), min(img.height, rect[3]+20)
                    if img.width > img.height:
                        # rotate
                        img_array = np.asarray(img)
                        left = np.sum(img_array[:, :int(img.width/2), :], (0, 1, 2))
                        right = np.sum(img_array[:, img.width-int(img.width/2):img.width, :], (0, 1, 2))
                        print(left, right)
                        if left > right:
                            img = img.transpose(Image.ROTATE_90)
                        else:
                            img = img.transpose(Image.ROTATE_270)
                    img = img.crop(rect)
                    img.save(str(simple_pic_path.joinpath(new_file_name)))
                    g.write(new_file_name + '\t' + text + '\n')


if __name__ == '__main__':
    preprocess()
