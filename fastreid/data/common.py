# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch.utils.data import Dataset
from .data_utils import read_image

import copy
import pdb

class CommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel

        name_set = set()
        gid_set = set()
        pid_set = set()
        cam_set = set()
        for i in img_items:
            name_set.add(i[0])
            gid_set.add(i[1])
            if isinstance(i[2], str): # training data
                pid_list = i[2][i[2].index('[')+1:i[2].index(']')].split(',')
                for x in pid_list:
                    if x[0] == '\'':
                        x = x[1:-1]
                    if x != '-1':
                        pid_set.add(x)
            if isinstance(i[2], list):
                pid_list = i[2]
                for x in pid_list:
                    if x != '-1':
                        pid_set.add(x)

            cam_set.add(i[3])

        self.names = sorted(list(name_set))
        self.gids = sorted(list(gid_set))
        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set))
        if relabel:
            self.name_dict = dict([(p, i) for i, p in enumerate(self.names)])
            self.gid_dict = dict([(p, i) for i, p in enumerate(self.gids)])
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])
        # num = self.__len__()
        # for index in range(num):
        #     try:
        #         self.__getitem__(index)
        #     except:
        #         print(self.img_items[index])
        #         pdb.set_trace()
        # print('all item right.')
        # print(f'common dataset info: num gid {len(self.gids)}, num pids {len(self.pids)}')
        # pdb.set_trace()

    def __len__(self):
        return len(self.img_items)


    def __getitem__(self, index):
        img_item = self.img_items[index]
        img_path = img_item[0]
        gid = img_item[1]

        pid = []
        if isinstance(img_item[2], str):  # training data
            pid_list = img_item[2][img_item[2].index('[') + 1:img_item[2].index(']')].split(',')
            for x in pid_list:
                if x[0] == '\'':
                    x = x[1:-1]
                if x != '-1':
                    pid.append(x)
        elif isinstance(img_item[2], list):
            pid_list = img_item[2]
            for x in pid_list:
                if x != '-1':
                    pid.append(x)


        camid = img_item[3]
        bbox = img_item[4]

        img = read_image(img_path)
        full_x = img.size[0]
        full_y = img.size[1]
        layout = []

        img_ps = []
        pids = []
        name_out, gid_out, camid_out = None, None, None

        for each_p in range(len(pid)):
            if pid[each_p] == '-1':
                continue
            if 'CUHK-SYSU' in img_path:
                img_p = img.crop((bbox[each_p][0], bbox[each_p][1],bbox[each_p][0] + bbox[each_p][2],
                                  bbox[each_p][1] + bbox[each_p][3]))
                avg_x = (bbox[each_p][0] + bbox[each_p][2]/2)/full_x
                avg_y = (bbox[each_p][1] + bbox[each_p][3]/2)/full_y
                layout.append(torch.tensor([avg_x, avg_y]))
            else:
                print('ERROR DATASET NAME in common.py !!')
                return ValueError('Must given a specfic dataset name for person cropping.')
            if self.transform is not None:
                img_p = self.transform(img_p)
                img_ps.append(img_p)

            if self.relabel:
                name_out = self.name_dict[img_path]
                gid_out = self.gid_dict[gid]
                pid_out = self.pid_dict[pid[each_p]]
                camid_out = self.cam_dict[camid]
                pids.append(pid_out)
            else:
                name_out = img_path
                gid_out = gid
                pids = []
                camid_out = camid

        if self.transform is not None:
            img = self.transform(img)


        num_p = len(img_ps)
        img_ps = torch.stack(img_ps, dim=0)
        pids = torch.tensor(pids)
        layout = torch.stack(layout, dim=0)


        return {
            "images": img,
            "targets": gid_out,
            "images_p": img_ps,
            "targets_p": pids,
            "camids": camid_out,
            "img_paths": name_out,
            "num_p": num_p,
            "layout": layout
        }

    @property
    def num_classes(self):
        return [len(self.gids), len(self.pids)]


    @property
    def num_cameras(self):
        return len(self.cams)
