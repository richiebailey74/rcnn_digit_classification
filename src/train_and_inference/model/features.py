import torchvision
import torch.nn as nn
import torch
from torchvision import ops
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        model = torchvision.models.resnet50(pretrained=True)
        req_layers = list(model.children())[:8]
        self.backbone_layers = nn.Sequential(*req_layers)
        for ind, param in enumerate(self.backbone_layers.named_parameters()):
            param[1].requires_grad = True

    def forward(self, image):
        return self.backbone_layers(image)


class ProposalModule(nn.Module):
    def __init__(self, in_size, hidden_dim=512, anchor_count=9, dropout_probability=0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout_probability)
        self.n_anchors = anchor_count
        self.reg_head = nn.Conv2d(hidden_dim, anchor_count * 4, kernel_size=1)
        self.conv1 = nn.Conv2d(in_size, hidden_dim, kernel_size=3, padding=1)
        self.conf_head = nn.Conv2d(hidden_dim, anchor_count, kernel_size=1)


    def yield_proposals(self, anchors, offsets):
        anchors = ops.box_convert(anchors, in_fmt='xyxy', out_fmt='cxcywh')

        proposals_ = torch.zeros_like(anchors)
        proposals_[:, 2] = anchors[:, 2] * torch.exp(offsets[:, 2])
        proposals_[:, 3] = anchors[:, 3] * torch.exp(offsets[:, 3])
        proposals_[:, 0] = anchors[:, 0] + offsets[:, 0] * anchors[:, 2]
        proposals_[:, 1] = anchors[:, 1] + offsets[:, 1] * anchors[:, 3]

        proposals = ops.box_convert(proposals_, in_fmt='cxcywh', out_fmt='xyxy')

        return proposals

    def forward(self, features, pos_anc_ind=None, neg_anc_ind=None, pos_anc_coords=None):
        if pos_anc_ind is None or neg_anc_ind is None or pos_anc_coords is None:
            mode = 'eval'
        else:
            mode = 'train'

        out = self.conv1(features)
        out = F.relu(self.dropout(out))

        offsets_regression = self.reg_head(out)
        confidences = self.conf_head(out)

        if mode == 'eval':
            return confidences, offsets_regression
        elif mode == 'train':
            positive_confidences = confidences.flatten()[pos_anc_ind]
            negative_confidences = confidences.flatten()[neg_anc_ind]
            offsets_pos = offsets_regression.contiguous().view(-1, 4)[pos_anc_ind]
            proposals = self.yield_proposals(pos_anc_coords, offsets_pos)

            return positive_confidences, negative_confidences, offsets_pos, proposals
