""" Vision Transformer (ViT) in PyTorch
A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929
The official jax code is released and available at https://github.com/google-research/vision_transformer
Status/TODO:
* Models updated to be compatible with official impl. Args added to support backward compat for old PyTorch weights.
* Weights ported from official jax impl for 384x384 base and small models, 16x16 and 32x32 patches.
* Trained (supervised on ImageNet-1k) my custom 'small' patch model to 77.9, 'base' to 79.4 top-1 with this code.
* Hopefully find time and GPUs for SSL or unsupervised pretraining on OpenImages w/ ImageNet fine-tune in future.
Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert
Hacked together by / Copyright 2020 Ross Wightman
"""

import logging
import math
import pdb
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastreid.layers import DropPath, trunc_normal_, to_2tuple
from fastreid.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from .build import BACKBONE_REGISTRY

logger = logging.getLogger(__name__)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            pdb.set_trace()
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed_overlap(nn.Module):
    """ Image to Patch Embedding with overlapping patches
    """

    def __init__(self, img_size=224, patch_size=16, stride_size=20, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape

        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)

        x = x.flatten(2).transpose(1, 2)  # [64, 8, 768]
        return x


class p_ViT(nn.Module):
    """ Vision Transformer
        A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
            - https://arxiv.org/abs/2010.11929
        Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
            - https://arxiv.org/abs/2012.12877
        """

    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., camera=0, drop_path_rate=0., hybrid_backbone=None,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), sie_xishu=1.0):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed_overlap(
                img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=in_chans,
                embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x


class g_ViT(nn.Module):
    """ Vision Transformer
        A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
            - https://arxiv.org/abs/2010.11929
        Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
            - https://arxiv.org/abs/2012.12877
        """

    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., camera=0, drop_path_rate=0., hybrid_backbone=None,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), sie_xishu=1.0):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))


        num_person = 20
        self.group_embed = nn.Parameter(torch.zeros(num_person, 1, embed_dim))
        trunc_normal_(self.group_embed, std=.02)

        self.sampling = 10
        self.group_embed_2D = nn.Parameter(torch.zeros(2, self.sampling, 384))
        trunc_normal_(self.group_embed_2D, std=.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.cls_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token', 'group_embed', 'group_embed_2D'}

    def member_uncertainty_modeling(self, x, t_member, c_member):
        P0 = torch.FloatTensor([0.5])
        p_max = torch.FloatTensor([0.3])
        sigma = p_max / (torch.FloatTensor([2]).sqrt() * torch.erfinv(1 - 2 * P0) + 3)
        mu = p_max - 3 * sigma

        if c_member <= torch.ceil(0.6 * t_member):
            drop_prob = torch.FloatTensor([0])
        elif c_member == t_member:
            drop_prob = nn.functional.relu(torch.normal(mu.item(), sigma.item(), (1,)))
        else:
            p_max_new = 1 - (1 - p_max) * t_member / c_member
            sigma_new = p_max_new / (torch.FloatTensor([2]).sqrt() * torch.erfinv(1 - 2 * P0) + 3)
            mu_new = p_max_new - 3 * sigma_new
            if p_max_new <= 0:
                drop_prob = torch.FloatTensor([0])
            else:
                drop_prob = nn.functional.relu(torch.normal(mu_new.item(), sigma_new.item(), (1,)))
        x = nn.functional.dropout2d(x, p=drop_prob.item(), training=self.training)
        return x

    def layout_uncertainty_modeling(self, ori_layout):
        if ori_layout.dtype is not torch.double:
            ori_layout = ori_layout.double()
        # special case for only one person in group
        if ori_layout.shape[1] == 1:
            shape = ori_layout.shape
            return torch.rand(shape).to(ori_layout.device)

        ori_layout = ori_layout.squeeze(0)
        # the shape of ori_layout: (N,2)
        N = ori_layout.shape[0]
        ones = torch.ones((N, 1)).to(ori_layout.device)
        ori_layout = torch.cat([ori_layout, ones], dim=1)

        Affine = torch.rand(3, 3) * 2 - 1
        Affine[2, :] = torch.tensor([0., 0., 1.])
        Affine = Affine.double().to(ori_layout.device)

        aff_layout = torch.mm(Affine, ori_layout.T)
        aff_layout = aff_layout[:2, :].T


        range_x = torch.rand(2).double().to(ori_layout.device).sort()[0]
        range_y = torch.rand(2).double().to(ori_layout.device).sort()[0]

        upper = torch.max(aff_layout, dim=0)[0]
        lower = torch.min(aff_layout, dim=0)[0]

        k1 = (range_x[1] - range_x[0]) / (upper[0] - lower[0])
        k2 = (range_y[1] - range_y[0]) / (upper[1] - lower[1])
        K = torch.diag(torch.tensor([k1, k2])).to(ori_layout.device)


        range_lower = torch.tensor([range_x[0], range_y[0]]).to(ori_layout.device)
        dst_layout = (aff_layout - lower) @ K + range_lower

        return dst_layout.unsqueeze(0)

    def feature_combine(self, appear, layout):
        layout_index = torch.floor(layout / (1 / self.sampling)).int()
        output = []
        for i in range(layout.shape[1]):
            index_x = layout_index[0, i, 0]
            index_y = layout_index[0, i, 1]
            # concat
            layout_feature = torch.cat([self.group_embed_2D[0, index_x, :],
                                        self.group_embed_2D[1, index_y, :]], dim=0)
            if appear[0, i, :].abs().sum() > 0:
                output.append(appear[0, i, :] + layout_feature)
        return torch.stack(output).unsqueeze(0)


    def forward(self, x, layout, t_member=None, c_member=None):

        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    
        if self.training:
            temp = self.member_uncertainty_modeling(x, t_member, c_member)
            x = temp if temp.sum() > 0 else x
            layout = self.layout_uncertainty_modeling(layout)

        x = self.feature_combine(x, layout)


        num_person = x.shape[1]
        x = torch.cat((cls_tokens, x), dim=1)
        x[0, 0, :] = x[0, 0, :] + self.group_embed[num_person]

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x


class GVit(nn.Module):
    """ Vision Transformer
        A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
            - https://arxiv.org/abs/2010.11929
        Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
            - https://arxiv.org/abs/2012.12877
        """

    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., camera=0, drop_path_rate=0., hybrid_backbone=None,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), sie_xishu=1.0):
        super().__init__()

        self.p_vit = p_ViT(img_size=img_size, sie_xishu=sie_xishu, stride_size=stride_size, depth=depth,
                           num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                           drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate)
        self.g_vit = g_ViT(img_size=img_size, sie_xishu=sie_xishu, stride_size=stride_size, depth=2,
                           num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                           drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate)



    def forward(self, imgs_g, imgs_p, layout, p_mask, n_t=None, n_c=None):
        feat_p = self.p_vit(imgs_p)
        feat_p_token = feat_p[:, 0].reshape(feat_p.shape[0], -1, 1, 1)

        feat_g_token = []
        for i in range(imgs_g.shape[0]):
            feat_p_temp = feat_p[:, 0][p_mask == i].unsqueeze(0)
            layout_temp = layout[p_mask == i].unsqueeze(0)

            if n_c == None:
                nc0 = feat_p_temp.shape[1]
                nt0 = nc0
            else:
                nt0 = n_t[i]
                nc0 = n_c[i]

            each_fea_g = self.g_vit(feat_p_temp, layout_temp, t_member=nt0, c_member=nc0)
            each_fea_g_token = each_fea_g[:, 0].reshape(each_fea_g.shape[0], -1, 1, 1)
            feat_g_token.append(each_fea_g_token)
        feat_g_token = torch.cat(feat_g_token, dim=0)


        return feat_g_token, feat_p_token


def resize_pos_embed(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))
    logger.info('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape,
                                                                                                      posemb_new.shape,
                                                                                                      hight,
                                                                                                      width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb


@BACKBONE_REGISTRY.register()
def build_gvit_backbone(cfg):
    """
    Create a Vision Transformer instance from config.
    Returns:
        SwinTransformer: a :class:`SwinTransformer` instance.
    """
    # fmt: off
    input_size = cfg.INPUT.SIZE_TRAIN
    pretrain = cfg.MODEL.BACKBONE.PRETRAIN
    pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
    depth = cfg.MODEL.BACKBONE.DEPTH
    sie_xishu = cfg.MODEL.BACKBONE.SIE_COE
    stride_size = cfg.MODEL.BACKBONE.STRIDE_SIZE
    drop_ratio = cfg.MODEL.BACKBONE.DROP_RATIO
    drop_path_ratio = cfg.MODEL.BACKBONE.DROP_PATH_RATIO
    attn_drop_rate = cfg.MODEL.BACKBONE.ATT_DROP_RATE
    # fmt: on

    num_depth = {
        'small': 8,
        'base': 12,
    }[depth]

    num_heads = {
        'small': 8,
        'base': 12,
    }[depth]

    mlp_ratio = {
        'small': 3.,
        'base': 4.
    }[depth]

    qkv_bias = {
        'small': False,
        'base': True
    }[depth]

    qk_scale = {
        'small': 768 ** -0.5,
        'base': None,
    }[depth]

    model = GVit(img_size=input_size, sie_xishu=sie_xishu, stride_size=stride_size, depth=num_depth,
                 num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                 drop_path_rate=drop_path_ratio, drop_rate=drop_ratio, attn_drop_rate=attn_drop_rate, )

    if pretrain:
        load_pretrain_model(pretrain_path, model.p_vit)
        # load_pretrain_model(pretrain_path, model.g_vit)
    return model


def load_pretrain_model(pretrain_path, model):
    try:
        state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
        logger.info(f"Loading pretrained model from {pretrain_path}")

        if 'model' in state_dict:
            state_dict = state_dict.pop('model')
        if 'state_dict' in state_dict:
            state_dict = state_dict.pop('state_dict')
        for k, v in state_dict.items():
            if 'head' in k or 'dist' in k:
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = model.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
                # To resize pos embedding when using model at different size from pretrained weights
                if 'distilled' in pretrain_path:
                    logger.info("distill need to choose right cls token in the pth.")
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v = resize_pos_embed(v, model.pos_embed.data, model.patch_embed.num_y, model.patch_embed.num_x)
            state_dict[k] = v
    except FileNotFoundError as e:
        logger.info(f'{pretrain_path} is not found! Please check this path.')
        raise e
    except KeyError as e:
        logger.info("State dict keys error! Please check the state dict.")
        raise e

    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        logger.info(
            get_missing_parameters_message(incompatible.missing_keys)
        )
    if incompatible.unexpected_keys:
        logger.info(
            get_unexpected_parameters_message(incompatible.unexpected_keys)
        )
