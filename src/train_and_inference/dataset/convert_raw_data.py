import h5py
import json
import tarfile
import requests
from time import time


def generate_annotations(data_split):
    with h5py.File(f"data/{data_split}/digitStruct.mat", 'r') as file:
        count = 1
        failures = []
        data = []
        for ind in range(len(file['digitStruct']['bbox'])):
            try:
                x_ref = file['digitStruct']['bbox'][ind][0]
                y_ref = file['digitStruct']['name'][ind][0]
                filename = ''.join(chr(c[0]) for c in file[y_ref])
                bbox = {}
                for attr in file[x_ref]:
                    # Since each attribute could potentially be an array of values
                    attr_data = file[x_ref][attr]
                    bbox[attr] = [file[attr_value[0]][0][0] for attr_value in attr_data]

                sample = dict()
                sample["image_path"] = f"{data_split}/{filename}"
                boxes = []
                labels = []
                for samp in range(len(bbox['label'])):
                    height, left, top, width = bbox["height"][samp], bbox["left"][samp], bbox["top"][samp], bbox["width"][
                        samp]
                    x_min = left
                    y_min = top
                    x_max = left + width
                    y_max = top + height
                    label = bbox["label"][samp]
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(label)
                sample["boxes"] = boxes
                sample["labels"] = labels
                data.append(sample)
            except Exception as e:
                failures.append(count)
                print(f"sample {count} failed")

            count += 1
            if count % 100 == 0:
                print(f"Data conversion progress at sample {count}")

        print(f"Number of failures for {data_split} is {len(failures)}")
        print(f"Number of successes for {data_split} is {len(data)}")
        with open(f'data/{data_split}_annotations.json', 'w') as json_file:
            json.dump(data, json_file)


def read_in_data_from_hf(url, save_path):
    ts = time()
    response = requests.get(url, stream=True)
    print(response)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.raw.read())
    else:
        print("Failed to download file")
        print(f"Failed to download file, status code: {response.status_code}")
        print("Response Body:", response.text)

    te = time()
    print(f"Time taken to read in data from HF is {te - ts}")


def convert_targz_to_readable(tar_path, output_path):
    with tarfile.open(tar_path, 'r:gz') as tar:
        # Extract all contents to the output directory
        tar.extractall(path=output_path)


def extract_transform_load_data(data_split):
    print("Commencing extracting targz from huggingface")
    url = 'https://huggingface.co/datasets/richiebailey/faster_rcnn_svhn_dataset/resolve/main/train.tar.gz'
    read_in_data_from_hf(url, f'{data_split}.tar.gz')
    print("Finishsed extracting targz from huggingface")
    print(f"Commencing targz extracting {data_split} data")
    convert_targz_to_readable(f"{data_split}.tar.gz", "data")
    print(f"Completed targz extracting {data_split} data")
    print(f"Commencing generating annotations for {data_split} data")
    generate_annotations(data_split)
    print(f"Completed generating annotations for {data_split} data")


def extract_data_loader(data_split):
    print("Commencing extracting data loader from huggingface")
    url = 'https://huggingface.co/datasets/richiebailey/faster_rcnn_svhn_dataset/resolve/main/train_data.dill'
    read_in_data_from_hf(url, f'{data_split}_dataset.dill')
    print("Finishsed extracting data loader from huggingface")
