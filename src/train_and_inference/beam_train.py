"""
> Run the API: beam serve app.py:predict
> Deploy it: beam deploy app.py:predict
"""
import torch
import torchvision
from beam import App, Runtime, Image, Volume
import json
from model import TwoStageDetector
from dataset import extract_data_loader
from torch.utils.data import DataLoader
import ssl
from time import time
import dill

"""
*** Beam commands
> deply app/code as a serverless REST API:
cd quickstart && beam deploy app.py

> call API (customize with app ID:
curl -X POST 'https://{YOUR_APP_ID}.apps.beam.cloud/' -H 'Authorization: Basic **** =' -H 'Content-Type: application/json' -d '{}'
"""





CACHE_PATH = "./cached_models"  # are we sure this is for beam or is for HF? does it even matter, who cares?


beam_app = App(
    name="faster_rcnn_training",
    runtime=Runtime(
        cpu=2,
        memory="32Gi",
        gpu="A100-80",
        image=Image(
            python_version="python3.9",
            python_packages=[
                "accelerate",
                "torch",
                "torchvision",
                "bitsandbytes",
                "huggingface_hub",
                "h5py",
                "matplotlib",
                "dill",
                "scipy",
                "numpy",
                "protobuf",
                "sentencepiece",
                "fastapi",
                "pydantic",
                "typing",
                "opencv-python",
                "psutil",
                "tqdm",
            ],  # You can also add a path to a requirements.txt instead
        ),
    ),
    # Storage volume for cached models
    volumes=[Volume(name="cached_models", path=CACHE_PATH)],
)


def training_loop(model, learning_rate, train_dataloader, n_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device used for training loop is", device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    loss_list = []

    for i in range(1):
        ts = time()
        total_loss = 0
        count = 0
        for img_batch, gt_bboxes_batch, gt_classes_batch in train_dataloader:
            img_batch = img_batch.to(device)
            gt_bboxes_batch = gt_bboxes_batch.to(device)
            gt_classes_batch = gt_classes_batch.to(device)

            # forward pass
            loss = model(img_batch, gt_bboxes_batch, gt_classes_batch)
            print(f"Loss for epoch {i} and iteration {count} is {loss}")

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1
            break

        loss_list.append(total_loss)

        te = time()
        print(f"Total time elapsed for epoch {i} is {te - ts}")

    torch.save(model.state_dict(), "saved/model.pt")
    return loss_list


def define_detector_and_loader(train_dataset):
    # get dataset and define the data loader for training
    dataset = train_dataset
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # pull resnet50 weights for the backbone
    ssl._create_default_https_context = ssl._create_unverified_context  # needed to do without error
    model = torchvision.models.resnet50(pretrained=True)
    req_layers = list(model.children())[:8]
    backbone = torch.nn.Sequential(*req_layers)

    # keep non-backbone weights not in frst four layers frozen
    for param in backbone.named_parameters():
        param[1].requires_grad = True

    # iterate once through loader in order to pull correct dimensionality
    for img_batch, gt_bboxes_batch, gt_classes_batch in data_loader:
        img_data_all = img_batch
        break
    img_data_all = img_data_all[:2]  # get first three dimensions (4th is number of samples, that doesn't matter)
    out = backbone(img_data_all)
    out_c, out_h, out_w = out.size(dim=1), out.size(dim=2), out.size(dim=3)

    # get variables to define the detector
    img_size = (224, 224)
    out_size = (out_h, out_w)
    n_classes = 11  # 10 digits and a none of them class
    roi_size = (2, 2)

    # define the detector
    detector = TwoStageDetector(img_size, out_size, out_c, n_classes, roi_size)

    # return detector and data loader
    return detector, data_loader


def load_training_data():
    data_split = "train"
    extract_data_loader(data_split)
    with open(f'{data_split}_dataset.dill', 'rb') as file:
        # Load the object from the file
        train_dataset = dill.load(file)
    return train_dataset


# Rest API initialized with loader and autoscaler
@beam_app.rest_api(
    loader=load_training_data,
    keep_warm_seconds=200,
)
def train(**inputs):
    print("inputs are", inputs.keys())  # eventually can make hyperparameters

    train_data = inputs["context"]

    detector, data_loader = define_detector_and_loader(train_data)

    learning_rate = 1e-3
    n_epochs = 200
    loss_list = training_loop(detector, learning_rate, data_loader, n_epochs)

    return {"return_message": "success", "losses": json.dumps(loss_list)}
