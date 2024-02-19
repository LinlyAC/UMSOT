# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import pdb

import torch
from torch import nn

from fastreid.config import configurable
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY

import numpy as np
import random
import copy

@META_ARCH_REGISTRY.register()
class Baseline(nn.Module):
    """
    Baseline architecture. Any models that contains the following two components:
    1. Per-image feature extraction (aka backbone)
    2. Per-image feature aggregation and loss computation
    """

    @configurable
    def __init__(
            self,
            *,
            backbone,
            heads_g,
            heads_p,
            pixel_mean,
            pixel_std,
            loss_kwargs=None
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone:
            heads:
            pixel_mean:
            pixel_std:
        """
        super().__init__()
        # backbone
        self.backbone = backbone

        # head
        self.heads_g = heads_g
        self.heads_p = heads_p

        self.loss_kwargs = loss_kwargs

        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(1, -1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(1, -1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        cfg0 = cfg.clone()
        if cfg0.is_frozen(): cfg0.defrost()
        backbone = build_backbone(cfg0)
        cfg0.MODEL.HEADS.NUM_CLASSES = cfg.MODEL.HEADS.NUM_CLASSES[0]

        heads_g = build_heads(cfg0)
        cfg0.MODEL.HEADS.NUM_CLASSES = cfg.MODEL.HEADS.NUM_CLASSES[1]
        cfg0.MODEL.HEADS.NAME = cfg.MODEL.HEADS.NAME
        cfg0.MODEL.BACKBONE.FEAT_DIM = cfg.MODEL.BACKBONE.FEAT_DIM
        heads_p = build_heads(cfg0)
        cfg0 = cfg.clone()

        return {
            'backbone': backbone,
            'heads_g': heads_g,
            'heads_p': heads_p,
            'pixel_mean': cfg.MODEL.PIXEL_MEAN,
            'pixel_std': cfg.MODEL.PIXEL_STD,
            'loss_kwargs':
                {
                    # loss name
                    'loss_names': cfg.MODEL.LOSSES.NAME,

                    # loss hyperparameters
                    'ce': {
                        'eps': cfg.MODEL.LOSSES.CE.EPSILON,
                        'alpha': cfg.MODEL.LOSSES.CE.ALPHA,
                        'scale': cfg.MODEL.LOSSES.CE.SCALE
                    },
                    'tri': {
                        'margin': cfg.MODEL.LOSSES.TRI.MARGIN,
                        'norm_feat': cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                        'hard_mining': cfg.MODEL.LOSSES.TRI.HARD_MINING,
                        'scale': cfg.MODEL.LOSSES.TRI.SCALE
                    },
                    'circle': {
                        'margin': cfg.MODEL.LOSSES.CIRCLE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.CIRCLE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.CIRCLE.SCALE
                    },
                    'cosface': {
                        'margin': cfg.MODEL.LOSSES.COSFACE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.COSFACE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.COSFACE.SCALE
                    },
                    'supcon':{

                    }
                }
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images, images_p, layout, p_mask = self.preprocess_image(batched_inputs)
        # pdb.set_trace()
        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"]
            targets_p = batched_inputs["targets_p"]
            n_t, n_c = self.counting_member_total(targets, p_mask)

            features, features_p = self.backbone(images, images_p, layout, p_mask, n_t, n_c)

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()
            if targets_p.sum() < 0: targets_p.zero_()
            outputs_g = self.heads_g(features, targets)
            outputs_p = self.heads_p(features_p, targets_p)

            losses = self.losses(outputs_g, outputs_p, targets, targets_p)
            return losses
        else:
            features, features_p = self.backbone(images, images_p, layout, p_mask)
            outputs_g = self.heads_g(features, None)
            return outputs_g


    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        # print('coming baseline.py, func() preprocess_image')
        # pdb.set_trace()
        images, images_p, p_mask = None, None, None
        if isinstance(batched_inputs, dict):
            images = batched_inputs['images']
            images_p = batched_inputs['images_p']
            layout = batched_inputs['layout']
            length = batched_inputs['images_p'].shape[0]
            p_mask = torch.zeros((length), dtype=torch.int)
            p_index, n_index, p_value = 0,0,0
            # pdb.set_trace()
            while p_index < length:
                p_mask[p_index : p_index + batched_inputs['num_p'][n_index]] = p_value
                p_index = p_index + batched_inputs['num_p'][n_index]
                n_index = n_index + 1
                p_value = p_value + 1
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))

        images.sub_(self.pixel_mean).div_(self.pixel_std)
        images_p.sub_(self.pixel_mean).div_(self.pixel_std)
        return images, images_p, layout, p_mask

    def counting_member_total(self, targets, p_mask):
        n_t = torch.zeros_like(targets)
        n_c = torch.zeros_like(targets)
        for target in targets.unique():
            target_indexs = torch.nonzero(targets == target).squeeze().cpu()
            temp = torch.tensor(-1)
            for j in target_indexs:
                n_c[j] = torch.nonzero(p_mask == j).numel()
                temp = max(temp, n_c[j])
            n_t[target_indexs] = temp
        return n_t.cpu(), n_c.cpu()


    def losses(self, outputs_g, outputs_p, gt_labels_g, gt_labels_p):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # model predictions
        # fmt: off
        pred_class_logits_g = outputs_g['pred_class_logits'].detach()
        cls_outputs_g       = outputs_g['cls_outputs']
        pred_features_g     = outputs_g['features']
        pred_class_logits_p = outputs_p['pred_class_logits'].detach()
        cls_outputs_p       = outputs_p['cls_outputs']
        pred_features_p     = outputs_p['features']
        # pred_class_logits_gc = outputs_gc['pred_class_logits'].detach()
        # cls_outputs_gc       = outputs_gc['cls_outputs']
        # pred_features_gc     = outputs_gc['features']
        # fmt: on

        # Log prediction accuracy

        log_accuracy(pred_class_logits_g, gt_labels_g)
        loss_dict = {}
        loss_names = self.loss_kwargs['loss_names']

        if 'CrossEntropyLoss' in loss_names:
            ce_kwargs = self.loss_kwargs.get('ce')
            loss_dict['loss_cls_g'] = cross_entropy_loss(
                cls_outputs_g,
                gt_labels_g,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale')

            loss_dict['loss_cls_p'] = cross_entropy_loss(
                cls_outputs_p,
                gt_labels_p,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale')

            # loss_dict['loss_cls_gc'] = cross_entropy_loss(
            #     cls_outputs_gc,
            #     gt_labels_g,
            #     ce_kwargs.get('eps'),
            #     ce_kwargs.get('alpha')
            # ) * ce_kwargs.get('scale')

        if 'TripletLoss' in loss_names:
            tri_kwargs = self.loss_kwargs.get('tri')
            loss_dict['loss_triplet_g'] = triplet_loss(
                pred_features_g,
                gt_labels_g,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale')

            loss_dict['loss_triplet_p'] = triplet_loss(
                pred_features_p,
                gt_labels_p,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale')

            # loss_dict['loss_triplet_gc'] = triplet_loss(
            #     pred_features_gc,
            #     gt_labels_g,
            #     tri_kwargs.get('margin'),
            #     tri_kwargs.get('norm_feat'),
            #     tri_kwargs.get('hard_mining')
            # ) * tri_kwargs.get('scale')

            # loss_dict['loss_kl'] = kl_loss(
            #     pred_features_g,
            #     pred_features_gc,
            # ) * 0.01

            # loss_dict['loss_mmd'] = self.mmd(
            #     pred_features_g,
            #     pred_features_gc,
            # ) * 1e1



        if 'CircleLoss' in loss_names:
            circle_kwargs = self.loss_kwargs.get('circle')
            loss_dict['loss_circle'] = pairwise_circleloss(
                pred_features,
                gt_labels,
                circle_kwargs.get('margin'),
                circle_kwargs.get('gamma')
            ) * circle_kwargs.get('scale')

        if 'Cosface' in loss_names:
            cosface_kwargs = self.loss_kwargs.get('cosface')
            loss_dict['loss_cosface'] = pairwise_cosface(
                pred_features,
                gt_labels,
                cosface_kwargs.get('margin'),
                cosface_kwargs.get('gamma'),
            ) * cosface_kwargs.get('scale')



        return loss_dict
