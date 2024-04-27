import torch.nn as nn
import torch
import torch.nn.functional as F
from .rpn import RegionProposalNetwork
from .classification import ClassificationModule


class TwoStageDetector(nn.Module):
    def __init__(self, image_size, output_size, channels_output_count, class_count, roi_size):
        super().__init__()
        self.object_classification = ClassificationModule(channels_output_count, class_count, roi_size)
        self.region_proposal_network = RegionProposalNetwork(image_size, output_size, channels_output_count)

    def forward(self, images, bounding_boxes, labels):
        combined_region_proposal_network_loss, feature_map, proposals, \
            ancor_index_positive, label_position = self.region_proposal_network(images, bounding_boxes, labels)

        # get separate proposals for each sample
        positive_proposals = []
        batch_size = images.size(dim=0)
        for ind in range(batch_size):
            proposal_indices = torch.where(ancor_index_positive == ind)[0]
            proposals_sep = proposals[proposal_indices].detach().clone()
            positive_proposals.append(proposals_sep)

        classification_loss = self.object_classification(feature_map, positive_proposals, label_position)
        total_loss = classification_loss + combined_region_proposal_network_loss

        return total_loss

    def inference(self, images, confidence_thresh=0.5, non_max_suppression_thresh=0.7):
        proposals, confidence_scores, features = self.region_proposal_network.inference(images, confidence_thresh, non_max_suppression_thresh)
        classification_scores = self.object_classification(features, proposals)
        class_probabilities = F.softmax(classification_scores, dim=-1)
        classes_all = torch.argmax(class_probabilities, dim=-1)

        classes_final = []
        count = 0
        batch_size = images.size(dim=0)
        for i in range(batch_size):
            proposal_count = len(proposals[i])
            classes_final.append(classes_all[count: count + proposal_count])
            count += proposal_count

        return proposals, confidence_scores, classes_final






# citations: got inspiration and assistance when stuck in referencing both of the following (this is a citation for everything in the model directory):
    # S. Ren, K. He, R. Girshick, J. Sun. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. Received from https://arxiv.org/abs/1506.01497
    # wingedasengan927 pytorch-tutorials. Received from https://github.com/wingedrasengan927/pytorch-tutorials/tree/master/Object%20Detection
