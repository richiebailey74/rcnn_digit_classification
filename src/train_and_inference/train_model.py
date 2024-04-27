from .dataset import SVHNDataset
from .model import TwoStageDetector
import torch
import torchvision
from torch.utils.data import DataLoader
import dill
import ssl
import matplotlib.pyplot as plt


def training_loop(model, learning_rate, train_dataloader, n_epochs, device):
    device = torch.device(device)  # Ensure the device is properly set from a parameter
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    loss_list = []

    for i in range(n_epochs):
        total_loss = 0
        count = 0
        for images, bounding_boxes, labels in train_dataloader:
            images = images.to(device)
            bounding_boxes = bounding_boxes.to(device)
            labels = labels.to(device)

            loss = model(images, bounding_boxes, labels)
            print(f"Loss for epoch {i} and iteration {count} is {loss}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

            if count % 50 == 0:
                break

        loss_list.append(total_loss)

    return loss_list


dataset = SVHNDataset(json_file='../data/train_annotations.json', root_dir='../data', transform=None)
data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

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

detector = TwoStageDetector(img_size, out_size, out_c, class_count, roi_size)

lr = 1e-3
epoch_count = 100
loss_list = training_loop(detector, lr, data_loader, epoch_count, device="cpu")

plt.plot(loss_list)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Combined Loss (Classifier and Regressor) for Training\nData on Digit Classification for the SVHN Dataset")
plt.savefig("saved/training_loss_svhn.png")

with open('saved/training_loss_svhn.dill', 'wb') as file:
    dill.dump(loss_list, file)

torch.save(detector.state_dict(), "saved/model.pt")
