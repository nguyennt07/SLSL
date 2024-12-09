import os
import glob
import csv
import pickle
import yaml

import numpy as np

with open('../config.yml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    POSE_SIZE = config['model']['pose_size']


def make_data(poses_path, sentences_path, save_path, mode):
    assert mode in ['train', 'test', 'val']

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    poses_path = os.path.join(poses_path, mode)
    annotation = os.path.join(sentences_path, f'{mode}.csv')

    poses_path = glob.glob(os.path.join(poses_path, '*.npz'))
    sentence_map = dict()
    with open(annotation, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            sentence_map[row['SENTENCE_NAME']] = row['SENTENCE']

    data = list()

    for pose in poses_path:
        pose_data = np.load(pose)
        pose_id = os.path.splitext(os.path.basename(pose))[0]
        sentence = sentence_map[pose_id]

        data.append({
            'pose': pose_data['keyp'].reshape(-1, POSE_SIZE * 2),
            'pose_confidence': pose_data['conf'].reshape(-1, POSE_SIZE),
            'sentence': sentence
        })

    save_filename = os.path.join(save_path, f'{mode}.pkl')
    with open(save_filename, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    make_data('./raw/poses/', './raw/sentences', './processed/', 'train')
