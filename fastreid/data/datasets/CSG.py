# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import pdb
import pickle
import os.path as osp
import re
import warnings

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class CSG(ImageDataset):
    _junk_pids = [0, -1]
    dataset_dir = 'CUHK-SYSU'
    dataset_name = "CSG"

    def __init__(self, root='datasets', **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = '/media/data1/zhangquan/documents/GroupReID-SOT/dataset'
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = osp.join(self.dataset_dir, 'images')
        self.label_dir = osp.join(self.dataset_dir, 'GReID_label')

        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')


        train = self.process_dir(self.data_dir, self.label_dir, 'train')
        query = self.process_dir(self.data_dir, self.label_dir, 'query')
        gallery = self.process_dir(self.data_dir, self.label_dir, 'gallery')

        super(CSG, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, label_path, type):
        if type == 'train':
            labels = osp.join(label_path, f'cuhk_{type}.pkl')
            labels = open(labels, 'rb')
            labels = pickle.load(labels)
        elif type == 'query':
            labels = osp.join(label_path, f'cuhk_test.pkl')
            labels = open(labels, 'rb')
            labels = pickle.load(labels)
        elif type == 'gallery':
            labels_1 = osp.join(label_path, f'cuhk_test.pkl')
            labels_1 = open(labels_1, 'rb')
            labels_1 = pickle.load(labels_1)
            labels_2 = osp.join(label_path, f'cuhk_gallery.pkl')
            labels_2 = open(labels_2, 'rb')
            labels_2 = pickle.load(labels_2)


            labels = [labels_1[i] + labels_2[i] for i in range(len(labels_1))]
            # pdb.set_trace()


        img_paths = [osp.join(dir_path, x) for x in labels[0]]

        data = []
        for img_path in img_paths:
            index = img_paths.index(img_path)
            # in CUHK_SYSU_Group dataset, person and group id start from 0.
            # we force it start from 1
            gid = labels[1][index] + 1
            pid = labels[2][index]
            bbox = labels[3][index]
            camid = -1
            assert gid >= 0
            # assert 1 <= camid <= 6
            if type == 'train':
                camid = 0
            elif type == 'query':
                camid = 1
            elif type == 'gallery':
                camid = 2

            if type == 'train':
                gid = self.dataset_name + "_" + str(gid)
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
                # spceial operator for this dataset
                pid = pid.replace(' ', '')
            data.append((img_path, gid, pid, camid, bbox))

        return data
