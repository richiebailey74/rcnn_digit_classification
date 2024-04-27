import torch.nn as nn
import torch
from torchvision import ops
import torch.nn.functional as F
from .features import FeatureExtractor, ProposalModule


class RegionProposalNetwork(nn.Module):
    def __init__(self, image_size, output_size, output_channel_number):
        super().__init__()

        self.negative_threshold = 0.3
        self.positive_threshold = 0.7
        self.w_reg = 5
        self.w_conf = 1
        self.anchor_scales = [2, 4, 6]
        self.anchor_ratios = [0.5, 1, 1.5]
        self.anchor_box_count = len(self.anchor_scales) * len(self.anchor_ratios)
        self.image_h, self.image_w = image_size
        self.output_height, self.output_width = output_size
        self.width_scale_factor = self.image_w // self.output_width
        self.height_scale_factor = self.image_h // self.output_height
        self.proposal_module = ProposalModule(output_channel_number, anchor_count=self.anchor_box_count)
        self.feature_extractor_module = FeatureExtractor()


    def forward(self, images, bounding_boxes, labels):
        batch_size = images.size(dim=0)
        features = self.feature_extractor_module(images)
        anchor_x_points, anchor_y_points = self.anchor_centers(out_size=(self.output_height, self.output_width))
        anchor_base = self.gen_anc_base(anchor_x_points, anchor_y_points, self.anchor_scales, self.anchor_ratios, (self.output_height, self.output_width))
        anchor_boxes_all = anchor_base.repeat(batch_size, 1, 1, 1, 1)
        projected_bounding_boxes = self.project_bounding_boxes(bounding_boxes, self.width_scale_factor, self.height_scale_factor, mode='p2a')

        anchor_indices_positive, anchor_indices_negative, confidence_scores, \
            offsets, positive_labels, anchor_coordinates_positive, \
            anchor_coordinates_negative, positive_anchor_indices = required_anchors(anchor_boxes_all, projected_bounding_boxes, labels)

        conf_scores_pos, conf_scores_neg, offsets_pos, proposals = self.proposal_module(features, anchor_indices_positive, \
                                                                                        anchor_indices_negative,
                                                                                        anchor_coordinates_positive)

        classification_loss = self.classification_loss(conf_scores_pos, conf_scores_neg, batch_size)
        regression_loss = self.bounding_box_regression_loss(offsets, offsets_pos, batch_size)
        combined_region_proposal_network_loss = self.w_conf * classification_loss + self.w_reg * regression_loss
        return combined_region_proposal_network_loss, features, proposals, positive_anchor_indices, positive_labels

    def project_bounding_boxes(self, bounding_boxes, width_scale_factor, height_scale_factor, mode='a2p'):
        assert mode in ['a2p', 'p2a']

        projected_bounding_boxes = bounding_boxes.clone().reshape(bounding_boxes.size(dim=0), -1, 4)
        invalid_bounding_box_mask = (projected_bounding_boxes == -1)  # indicating padded bboxes

        if mode != 'a2p':
            projected_bounding_boxes[:, :, [0, 2]] /= width_scale_factor
            projected_bounding_boxes[:, :, [1, 3]] /= height_scale_factor

        else:
            projected_bounding_boxes[:, :, [0, 2]] *= width_scale_factor
            projected_bounding_boxes[:, :, [1, 3]] *= height_scale_factor

        projected_bounding_boxes.masked_fill_(invalid_bounding_box_mask, -1)  # fill padded bboxes back with -1
        projected_bounding_boxes.resize_as_(bounding_boxes)

        return projected_bounding_boxes

    def classification_loss(self, positive_confidence_scores, negative_confidence_scores, batch_size):
        negative_targets = torch.zeros_like(negative_confidence_scores)
        positive_targets = torch.ones_like(positive_confidence_scores)

        combined_targets = torch.cat((positive_targets, negative_targets))
        inputs = torch.cat((positive_confidence_scores, negative_confidence_scores))

        binary_cross_entropy_loss = F.binary_cross_entropy_with_logits(inputs, combined_targets, reduction='sum') * 1. / batch_size

        return binary_cross_entropy_loss

    def bounding_box_regression_loss(self, offsets, positive_offsets, batch_size):
        assert offsets.size() == positive_offsets.size()
        smooth_l1_loss = F.smooth_l1_loss(positive_offsets, offsets, reduction='sum') * 1. / batch_size
        return smooth_l1_loss

    def inference(self, images, confidence_threshold=0.5, non_maximum_suppression_threshold=0.7):
        with torch.no_grad():
            batch_size = images.size(dim=0)
            features = self.feature_extractor_module(images)

            anchor_x_points, anchor_y_points = self.anchor_centers(out_size=(self.output_height, self.output_width))
            anchor_base = self.gen_anc_base(anchor_x_points, anchor_y_points, self.anchor_scales, self.anchor_ratios,
                                            (self.output_height, self.output_width))
            anchor_boxes_all = anchor_base.repeat(batch_size, 1, 1, 1, 1)
            anchor_boxes_flattened = anchor_boxes_all.reshape(batch_size, -1, 4)

            predicted_confidence_scores, predicted_offsets = self.proposal_module(features)
            predicted_confidence_scores = predicted_confidence_scores.reshape(batch_size, -1)
            predicted_offsets = predicted_offsets.reshape(batch_size, -1, 4)

            proposals_final = []
            confidence_scores_final = []
            for i in range(batch_size):
                confidence_scores_current = torch.sigmoid(predicted_confidence_scores[i])
                offsets = predicted_offsets[i]
                anchor_boxes = anchor_boxes_flattened[i]
                proposals = self.yield_proposals(anchor_boxes, offsets)
                confidence_index = torch.where(confidence_scores_current >= confidence_threshold)[0]
                positive_confidence_scores = confidence_scores_current[confidence_index]
                positive_proposals = proposals[confidence_index]
                non_maximum_suppression_index = ops.nms(positive_proposals, positive_confidence_scores, non_maximum_suppression_threshold)
                positive_confidence_scores = positive_confidence_scores[non_maximum_suppression_index]
                positive_proposals = positive_proposals[non_maximum_suppression_index]
                proposals_final.append(positive_proposals)
                confidence_scores_final.append(positive_confidence_scores)

        return proposals_final, confidence_scores_final, features

    def anchor_centers(self, out_size):
        h, w = out_size

        anchor_x_points = torch.arange(0, w) + 0.5
        anchor_y_points = torch.arange(0, h) + 0.5

        return anchor_x_points, anchor_y_points

    def gen_anc_base(self, anchor_x_points, anchor_y_points, anchor_scales, anchor_ratios, out_size):
        anchor_box_count = len(anchor_scales) * len(anchor_ratios)
        anchor_base = torch.zeros(1, anchor_x_points.size(dim=0) \
                               , anchor_y_points.size(dim=0), anchor_box_count, 4)  # shape - [1, Hmap, Wmap, n_anchor_boxes, 4]

        for u, w in enumerate(anchor_y_points):
            for v, z in enumerate(anchor_x_points):
                c = 0
                anchor_boxes = torch.zeros((anchor_box_count, 4))
                for j, ratio in enumerate(anchor_ratios):
                    for i, scale in enumerate(anchor_scales):

                        x_min = z - (scale * ratio) / 2
                        y_min = w - scale / 2
                        x_max = z + (scale * ratio) / 2
                        y_max = w + scale / 2

                        anchor_boxes[c, :] = torch.Tensor([x_min, y_min, x_max, y_max])
                        c += 1

                anchor_base[:, v, u, :] = ops.clip_boxes_to_image(anchor_boxes, size=out_size)

        return anchor_base

    def yield_proposals(self, anchors, offsets):
        anchors = ops.box_convert(anchors, in_fmt='xyxy', out_fmt='cxcywh')

        proposals_ = torch.zeros_like(anchors)
        proposals_[:, 2] = anchors[:, 2] * torch.exp(offsets[:, 2])
        proposals_[:, 3] = anchors[:, 3] * torch.exp(offsets[:, 3])
        proposals_[:, 0] = anchors[:, 0] + offsets[:, 0] * anchors[:, 2]
        proposals_[:, 1] = anchors[:, 1] + offsets[:, 1] * anchors[:, 3]

        proposals = ops.box_convert(proposals_, in_fmt='cxcywh', out_fmt='xyxy')

        return proposals


# citation: this method is heavily inspired by wingedrasengan927 (I was stuck and couldn't figure out how to do it)
def required_anchors(anchor_boxes, bounding_boxes, labels, positive_threshold=0.7, negative_thresh=0.2):
    N = bounding_boxes.shape[1]
    B, W, H, A, _ = anchor_boxes.shape

    anchor_box_total = A * W * H
    iou_mat = get_iou_mat(B, anchor_boxes, bounding_boxes)
    max_iou_per_bounding_box, _ = iou_mat.max(dim=1, keepdim=True)

    positive_anchor_mask = torch.logical_and(iou_mat == max_iou_per_bounding_box, max_iou_per_bounding_box > 0)
    positive_anchor_mask = torch.logical_or(positive_anchor_mask, iou_mat > positive_threshold)
    positive_anchor_indices = torch.where(positive_anchor_mask)[0]
    positive_anchor_mask = positive_anchor_mask.flatten(start_dim=0, end_dim=1)
    positive_anc_index = torch.where(positive_anchor_mask)[0]

    max_iou_per_anchor, max_iou_per_anchor_index = iou_mat.max(dim=-1)
    max_iou_per_anchor = max_iou_per_anchor.flatten(start_dim=0, end_dim=1)

    confidence_scores = max_iou_per_anchor[positive_anc_index]

    labels_expanded = labels.view(B, 1, N).expand(B, anchor_box_total, N)
    classes = torch.gather(labels_expanded, -1, max_iou_per_anchor_index.unsqueeze(-1)).squeeze(-1)
    classes = classes.flatten(start_dim=0, end_dim=1)
    classes_positive = classes[positive_anc_index]

    bounding_boxes_expanded = bounding_boxes.view(B, 1, N, 4).expand(B, anchor_box_total, N, 4)
    bounding_boxes_ = torch.gather(bounding_boxes_expanded, -2,
                             max_iou_per_anchor_index.reshape(B, anchor_box_total, 1, 1).repeat(1, 1, 1, 4))
    bounding_boxes_ = bounding_boxes_.flatten(start_dim=0, end_dim=2)
    positive_bounding_boxes = bounding_boxes_[positive_anc_index]
    flattened_anchor_boxes = anchor_boxes.flatten(start_dim=0, end_dim=-2)
    positive_anchor_coordinates = flattened_anchor_boxes[positive_anc_index]
    offsets = calculate_offsets(positive_anchor_coordinates, positive_bounding_boxes)

    negative_anchor_mask = (max_iou_per_anchor < negative_thresh)
    negative_anchor_ind = torch.where(negative_anchor_mask)[0]
    negative_anchor_ind = negative_anchor_ind[torch.randint(0, negative_anchor_ind.shape[0], (positive_anc_index.shape[0],))]
    negative_anchor_coords = flattened_anchor_boxes[negative_anchor_ind]

    return positive_anc_index, negative_anchor_ind, confidence_scores, offsets, classes_positive, \
        positive_anchor_coordinates, negative_anchor_coords, positive_anchor_indices


def get_iou_mat(batch_size, anchor_boxes, bounding_boxes):
    flattened_anchor_boxes = anchor_boxes.reshape(batch_size, -1, 4)
    anchor_boxes_total = flattened_anchor_boxes.size(dim=1)
    ious_mat = torch.zeros((batch_size, anchor_boxes_total, bounding_boxes.size(dim=1)))

    for i in range(batch_size):
        gt_bboxes = bounding_boxes[i]
        anc_boxes = flattened_anchor_boxes[i]
        ious_mat[i, :] = ops.box_iou(anc_boxes, gt_bboxes)

    return ious_mat


def calculate_offsets(positive_anchor_coordinates, bounding_boxes_map):
    bounding_boxes_map = ops.box_convert(bounding_boxes_map, in_fmt='xyxy', out_fmt='cxcywh')
    positive_anchor_coordinates = ops.box_convert(positive_anchor_coordinates, in_fmt='xyxy', out_fmt='cxcywh')

    cx, cy, w, h = (bounding_boxes_map[:, 0], bounding_boxes_map[:, 1],
                    bounding_boxes_map[:, 2], bounding_boxes_map[:,3])
    cx_anchor, cy_anchor, w_anchor, h_anchor = (positive_anchor_coordinates[:, 0], positive_anchor_coordinates[:, 1],
                                    positive_anchor_coordinates[:, 2], positive_anchor_coordinates[:,3])

    th_ = torch.log(h / h_anchor)
    tw_ = torch.log(w / w_anchor)
    ty_ = (cy - cy_anchor) / h_anchor
    tx_ = (cx - cx_anchor) / w_anchor

    return torch.stack([tx_, ty_, tw_, th_], dim=-1)
