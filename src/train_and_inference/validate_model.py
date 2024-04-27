from .dataset import SVHNDataset
from .model import TwoStageDetector
import torch
import torchvision
from torchvision.ops import nms
from torch.utils.data import DataLoader
import dill
import ssl
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def test_loop(model, test_dataloader, device):
    device = torch.device(device)  # Ensure the device is properly set from a parameter
    model = model.to(device)
    model.eval()

    loss_list = []
    all_labels = []
    all_preds = []
    count = 0

    with torch.no_grad():  # No need to track gradients
        for images, bounding_boxes, labels in test_dataloader:
            print("uh")
            images = images.to(device)
            bounding_boxes = bounding_boxes.to(device)
            labels = labels.to(device)

            # Compute the loss using the forward pass
            total_loss = model(images, bounding_boxes, labels)
            loss_list.append(total_loss.item())
            print(f"Loss for batch {count} is {total_loss.item()}")

            # Perform inference to get predictions
            proposals, conf_scores, classes_final = model.inference(images)

            valid_classes_mask = labels != -1
            num_valid_classes = valid_classes_mask.sum().item()

            # Apply Non-Maximum Suppression
            keep_indices = nms(proposals[0], conf_scores[0], iou_threshold=0.5)
            top_k_indices = torch.topk(conf_scores[0][keep_indices], k=num_valid_classes, largest=True,
                                       sorted=True).indices
            final_indices = keep_indices[top_k_indices]
            filtered_preds = classes_final[0][final_indices].numpy()
            filtered_labels = labels[valid_classes_mask].numpy()

            # Collect predictions and actual labels for accuracy calculation
            print('Filtered preds', filtered_preds)
            all_preds.extend(filtered_preds)
            print("filtered labels", filtered_labels)
            all_labels.extend(filtered_labels)

            count += 1
            if count % 250 == 0:
                break

        # Calculate accuracy
        print("all labels", all_labels)
        print("all preds", all_preds)
        accuracy = accuracy_score(all_labels, all_preds)
        print(f"Accuracy: {accuracy}")

    return loss_list, accuracy


dataset = SVHNDataset(json_file='../data/test_annotations.json', root_dir='../data', transform=None)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

for images, bounding_boxes, labels in data_loader:
    images_ = images
    bounding_boxes_ = bounding_boxes
    labels_ = labels
    break
images_ = images_[:2]
bounding_boxes_ = bounding_boxes_[:2]
labels_ = labels_[:2]

ssl._create_default_https_context = ssl._create_unverified_context
model = torchvision.models.resnet50(pretrained=True).to("cpu")
layers_of_interest = list(model.children())[:8]
backbone = torch.nn.Sequential(*layers_of_interest)
for param in backbone.named_parameters():
    param[1].requires_grad = True
out = backbone(images_.to("cpu"))

out_c, out_h, out_w = out.size(dim=1), out.size(dim=2), out.size(dim=3)
w = 224
h = 224
w_scale_factor = w // out_w
h_scale_factor = h // out_h

img_size = (224, 224)
out_size = (out_h, out_w)
class_count = 11
roi_size = (2, 2)

sd = torch.load("saved/model.pt")
detector = TwoStageDetector(img_size, out_size, out_c, class_count, roi_size)
detector.load_state_dict(sd)

lr = 1e-3
epoch_count = 100
loss_list, accuracy = test_loop(detector, data_loader, "cpu")

plt.plot(loss_list)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Combined Loss (Classifier and Regressor) for Testing\nData on Digit Classification for the SVHN Dataset")
plt.savefig("saved/testing_loss_svhn.png")

with open('saved/testing_loss_svhn.dill', 'wb') as file:
    dill.dump(loss_list, file)

