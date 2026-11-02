"""
The backbone implementation is inspired by TransReID series. The pretrained weighted is provided by LUPerson.
Thanks for their excellent works!
TransReID: https://github.com/damo-cv/TransReID
TransReID-SSL: https://github.com/damo-cv/TransReID-SSL
LUPerson: https://github.com/DengpanFu/LUPerson
"""

import sys
import os.path as osp
sys.path.append(osp.abspath(osp.join(__file__, '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from libs.models.vit import vit_small_patch16_224_TransReID


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class TMGF(nn.Module):
    """
    Transformer-based Multi-Grained Feature encoder.
    """

    __factory = {
        'tmgf': vit_small_patch16_224_TransReID
    }

    def __init__(self, arch, img_size, sie_coef, camera_num, view_num, stride_size, drop_path_rate, drop_rate, attn_drop_rate,
                 pretrain_path, hw_ratio, gem_pool, stem_conv, num_parts, has_early_feature, has_head, global_feature_type,
                 granularities, branch, enable_early_norm, **kwargs):
        super().__init__()
        print(f'using Transformer_type: {arch} as a backbone')

        assert sum(granularities) == num_parts
        assert branch == 'all'
        assert has_head
        assert global_feature_type == 'mean'

        if camera_num:
            camera_num = camera_num
        else:
            camera_num = 0
        if view_num:
            view_num = view_num
        else:
            view_num = 0

        self.base = TMGF.__factory[arch](img_size=img_size, sie_xishu=sie_coef, camera=camera_num, view=view_num, stride_size=stride_size,
                                           drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                           gem_pool=gem_pool, stem_conv=stem_conv, has_early_feature=has_early_feature,
                                           enable_early_norm=enable_early_norm, **kwargs)
        self.in_planes = self.base.in_planes
        self.granularities = granularities

        if pretrain_path != '':
            if osp.exists(pretrain_path):
                self.base.load_param(pretrain_path, hw_ratio)
                print('Loading pretrained weights from {} ...'.format(pretrain_path))
            else:
                raise FileNotFoundError('Cannot find {}'.format(pretrain_path))
        else:
            print('Initialize weights randomly.')

        # Part split settings
        self.num_parts = num_parts
        self.num_parts_b1 = self.granularities[0]
        self.num_parts_b2 = self.granularities[1]
        self.fmap_h = img_size[0] // stride_size[0]
        self.fmap_w = img_size[1] // stride_size[1]

        # Two different granularity branches
        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(copy.deepcopy(block), copy.deepcopy(layer_norm))
        self.b2 = nn.Sequential(copy.deepcopy(block), copy.deepcopy(layer_norm))

        # Pooling layers
        for i, g in enumerate(self.granularities):
            setattr(self, 'b{}_pool'.format(i+1), nn.AvgPool2d(kernel_size=(self.fmap_h//g, self.fmap_w),
                                                               stride=(self.fmap_h//g,)))

        print('num_parts={}, branch_parts={}'.format(self.num_parts, self.granularities))

        # Global bottleneck
        self.bottleneck = self.make_bnneck(self.in_planes, weights_init_kaiming)

        # Part bottleneck
        self.part_bns = nn.ModuleList([
            self.make_bnneck(self.in_planes, weights_init_kaiming) for i in range(self.num_parts)
        ])

        self.offset_net_b1 = self.make_offset_net(self.num_parts_b1)
        self.offset_net_b2 = self.make_offset_net(self.num_parts_b2)
        self.mfe_score_b1 = self.make_mfe_score_net()
        self.mfe_score_b2 = self.make_mfe_score_net()
        self.asb_fusion_logit = nn.Parameter(torch.zeros(1))
        self.mfe_fusion_logit = nn.Parameter(torch.zeros(1))

    def get_fusion_weight(self, name):
        logit = getattr(self, '{}_fusion_logit'.format(name))
        return 0.2 * torch.sigmoid(logit)

    def make_offset_net(self, num_parts):
        return nn.Sequential(
            nn.Conv2d(self.in_planes, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_parts * 2, kernel_size=1)
        )

    def make_mfe_score_net(self):
        hidden_dims = max(self.in_planes // 4, 64)
        score_net = nn.Sequential(
            nn.Linear(self.in_planes * 4, hidden_dims),
            nn.LayerNorm(hidden_dims),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims, 4)
        )
        weights_init_kaiming(score_net[0])
        nn.init.normal_(score_net[3].weight, std=0.01)
        nn.init.constant_(score_net[3].bias, 0.0)
        return score_net

    def forward_multi_branch(self, x, label=None, cam_label=None, view_label=None):
        """
        ViT 1st ~ (L-1)-th layers, duplicated L-th layers as projection heads for two branches.
        """

        x = self.base(x, cam_label=cam_label, view_label=view_label) # output before last layer
        B = x.size(0)

        # Split after head
        # branch 1
        x_b1 = self.b1(x) # (B, L, C)
        x_b1_glb = x_b1[:, 0, :] # (B, C)
        x_b1_patch = x_b1[:, 1:, :] # (B, L-1, C)
        x_b1_patch = x_b1_patch.permute(0, 2, 1).reshape((B, self.in_planes, self.fmap_h, self.fmap_w))
        x_b1_part = self.b1_pool(x_b1_patch).squeeze(-1) # (B, C, P1)

        x_b1_dyn = self.asb_pool(x_b1_patch, self.num_parts_b1, self.offset_net_b1)
        x_b1_mfe = self.mfe_pool(x_b1_patch, self.num_parts_b1, self.mfe_score_b1)
        x_b1_part = (x_b1_part + self.get_fusion_weight('asb') * x_b1_dyn
                     + self.get_fusion_weight('mfe') * x_b1_mfe)

        # branch 2
        x_b2 = self.b2(x)
        x_b2_glb = x_b2[:, 0, :]
        x_b2_patch = x_b2[:, 1:, :]
        x_b2_patch = x_b2_patch.permute(0, 2, 1).reshape((B, self.in_planes, self.fmap_h, self.fmap_w))
        x_b2_part = self.b2_pool(x_b2_patch).squeeze(-1) # (B, C, P2)

        x_b2_dyn = self.asb_pool(x_b2_patch, self.num_parts_b2, self.offset_net_b2)
        x_b2_mfe = self.mfe_pool(x_b2_patch, self.num_parts_b2, self.mfe_score_b2)
        x_b2_part = (x_b2_part + self.get_fusion_weight('asb') * x_b2_dyn
                     + self.get_fusion_weight('mfe') * x_b2_mfe)

        # Mean global feature
        x_glb = 0.5 * (x_b1_glb + x_b2_glb) # (B, C)

        # Stack two branch part features
        x_part = torch.cat([x_b1_part, x_b2_part], dim=2) # (B, C, P), P = P1 + P2

        return x_glb, x_part

    def asb_pool(self, x_patch, num_parts, offset_net):
        B = x_patch.size(0)
        offsets = offset_net(x_patch).mean(dim=(2, 3)).view(B, num_parts, 2)
        grid = self._generate_dynamic_grid(B, num_parts, offsets, x_patch.device)
        return F.grid_sample(x_patch, grid, align_corners=False).squeeze(-1)

    def mfe_pool(self, x_patch, num_parts, score_layer):
        cell_h = self.fmap_h // num_parts // 2
        cell_w = self.fmap_w // 2
        B, C, H, W = x_patch.shape
        expected_h = num_parts * 2 * cell_h
        expected_w = 2 * cell_w
        if H != expected_h or W != expected_w:
            raise RuntimeError('MFE expects feature map {}, got {}'.format((expected_h, expected_w), (H, W)))

        cell_inputs = x_patch.reshape(B, C, num_parts * 2, cell_h, 2, cell_w)
        cell_inputs = cell_inputs.permute(0, 2, 4, 1, 3, 5).contiguous()
        cell_inputs = cell_inputs.reshape(B * num_parts * 4, C, cell_h, cell_w)

        cell_desc = F.adaptive_avg_pool2d(cell_inputs, output_size=1).view(B, num_parts, 4, C)
        fine_cells = cell_desc.permute(0, 3, 1, 2).contiguous()

        part_desc = cell_desc.reshape(B * num_parts, 4 * C)
        scores = score_layer(part_desc).reshape(B, num_parts, 4).unsqueeze(1)
        weights = F.softmax(scores, dim=-1)
        return (fine_cells * weights).sum(dim=-1)

    def _generate_dynamic_grid(self, B, P, offsets, device):
        initial_y = torch.linspace(-1 + 1 / P, 1 - 1 / P, P, device=device)
        initial_x = torch.zeros(P, device=device)
        base_grid = torch.stack([initial_x, initial_y], dim=1).unsqueeze(0).expand(B, P, 2)

        part_id = torch.arange(P, device=device)
        y_min = -1 + 2 * part_id / P + 1 / self.fmap_h
        y_max = -1 + 2 * (part_id + 1) / P - 1 / self.fmap_h
        lower = torch.stack([-torch.ones_like(y_min), y_min], dim=1)
        upper = torch.stack([torch.ones_like(y_max), y_max], dim=1)

        dynamic_grid = torch.maximum(
            torch.minimum(base_grid + offsets, upper.unsqueeze(0)),
            lower.unsqueeze(0)
        )
        return dynamic_grid.unsqueeze(2)

    def forward(self, x, label=None, cam_label=None, view_label=None):
        B = x.size(0)
        x_glb, x_part = self.forward_multi_branch(x, label, cam_label, view_label)

        # BNNeck + L2 norm
        x_glb = self.bottleneck(x_glb)
        x_part = torch.stack([self.part_bns[i](x_part[:, :, i]) for i in range(x_part.size(2))], dim=2)

        x_glb = F.normalize(x_glb, dim=1)
        x_part = F.normalize(x_part, dim=1)

        assert x_part.size(2) == self.num_parts, 'x_part size: {} != num_parts: {}'.format(
            x_part.size(2), self.num_parts) # check part num

        return {'global': x_glb, 'part': x_part.permute(2, 0, 1)} # x_part as (P, B, C)

    def make_bnneck(self, dims, init_func):
        bn = nn.BatchNorm1d(dims)
        bn.bias.requires_grad_(False) # disable bias update
        bn.apply(init_func)
        return bn


def tmgf(**kwargs):
    return TMGF(**kwargs)
