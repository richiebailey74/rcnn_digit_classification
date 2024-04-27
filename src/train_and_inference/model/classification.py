import torch.nn as nn
from torchvision import ops
import torch.nn.functional as F


class ClassificationModule(nn.Module):
    def __init__(self, out_size, class_count, roi_size, hidden_dim=512, dropout_probability=0.3):
        super().__init__()
        self.cls_head = nn.Linear(hidden_dim, class_count)
        self.dropout = nn.Dropout(dropout_probability)
        self.roi_size = roi_size
        self.fc = nn.Linear(out_size, hidden_dim)
        self.avg_pool = nn.AvgPool2d(self.roi_size)

    def forward(self, features, proposals, classes=None):
        if classes is not None:
            mode = 'train'
        else:
            mode = 'eval'

        roi_out = ops.roi_pool(features, proposals, self.roi_size)
        roi_out = self.avg_pool(roi_out)
        roi_out = roi_out.squeeze(-1).squeeze(-1)
        out = self.fc(roi_out)
        out = F.relu(self.dropout(out))
        cls_scores = self.cls_head(out)

        if mode == 'train':
            cls_loss = F.cross_entropy(cls_scores, classes.long())
            return cls_loss

        return cls_scores
