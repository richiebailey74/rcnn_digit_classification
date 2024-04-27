# TODO: this is the driver file that the TA's will run in order to grade the assignment (put in the 5 images)...
# TODO: make sure the requirements.txt install all of the necessary requirements to be able to run this o a CPU without
# TODO: any errors being thrown
from final_project.src.train_and_inference.dataset import SVHNDataset
from final_project.src.train_and_inference.model import TwoStageDetector
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.ops import nms
from sklearn.metrics import accuracy_score
import ssl
from torchvision.transforms.functional import to_pil_image
from PIL import Image


if __name__ == '__main__':
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
                images = images.to(device)
                bounding_boxes = bounding_boxes.to(device)
                labels = labels.to(device)

                # Compute the loss using the forward pass
                total_loss = model(images, bounding_boxes, labels)
                loss_list.append(total_loss.item())

                # Perform inference to get predictions
                proposals, conf_scores, classes_final = model.inference(images)

                valid_classes_mask = labels != -1
                filtered_labels = labels[valid_classes_mask].numpy()
                num_valid_classes = valid_classes_mask.sum().item()

                # Apply Non-Maximum Suppression
                keep_indices = nms(proposals[0], conf_scores[0], iou_threshold=0.4)
                top_k_indices = torch.topk(conf_scores[0][keep_indices], k=num_valid_classes, largest=True,
                                           sorted=True).indices
                final_indices = keep_indices[top_k_indices]
                filtered_preds = classes_final[0][final_indices].numpy()


                # Collect predictions and actual labels for accuracy calculation
                print('Ground Truth Labels', filtered_labels)
                all_labels.extend(filtered_labels)
                print("Predicted Labels", filtered_preds)
                all_preds.extend(filtered_preds)

                count += 1
                save_tensor_as_png(images[0], f"graded_images/{count}.png")
                if count == 5:
                    break

            # Calculate accuracy
            print("all labels", all_labels)
            print("all preds", all_preds)
            accuracy = accuracy_score(all_labels, all_preds)
            print(f"Accuracy: {accuracy}")

        return loss_list


    def save_tensor_as_png(image_tensor, file_path):
        # Check if the tensor is in the range 0-1 and scale if necessary
        if image_tensor.max() <= 1:
            image_tensor = image_tensor * 255

        # Convert to uint8 if necessary
        if image_tensor.dtype != torch.uint8:
            image_tensor = image_tensor.to(torch.uint8)

        # Convert the tensor to a PIL Image
        image = to_pil_image(image_tensor)

        # Save the PIL Image
        image.save(file_path, format='PNG')

    print("Pulling data")
    dataset = SVHNDataset(json_file='../data/test_annotations.json', root_dir='../data', transform=None)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    print("Data pulled")

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

    sd = torch.load("../train_and_inference/saved/model.pt")
    detector = TwoStageDetector(img_size, out_size, out_c, class_count, roi_size)
    detector.load_state_dict(sd)

    lr = 1e-3
    epoch_count = 5
    loss_list = test_loop(detector, data_loader, "cpu")

