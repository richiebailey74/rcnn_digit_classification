import base64
import requests
import json
import matplotlib.pyplot as plt


# run the beam cloud API (while filling in the correct ID in the API key to run the training job)
def btoa(s: str) -> str:
    """
    Encodes a string in base-64 using Python, similar to TypeScript's btoa().

    :param s: The input string to be encoded.
    :return: The base-64 encoded string.
    """
    # Convert the input string to bytes
    string_bytes = s.encode('utf-8')

    # Encode the bytes in base-64
    base64_bytes = base64.b64encode(string_bytes)

    # Convert the base-64 encoded bytes back to a string
    base64_string = base64_bytes.decode('utf-8')

    return base64_string


if __name__ == '__main__':
    # TODO: put these things into a secrets file
    beam_client_id = ''
    beam_client_secret = ''
    btoa_auth = btoa(beam_client_id + ':' + beam_client_secret)
    app_id = ''
    url = f"https://{app_id}.apps.beam.cloud/"
    headers = {
        "Authorization": f"Basic {btoa_auth}=",
        "Content-Type": "application/json",
    }

    # TODO: could maybe pass hyperparameters for training here
    data = {"Empty": "Empty"}

    response = requests.post(url, headers=headers, json=data).json()

    print("response", response)

    training_losses = json.loads(response['losses'])

    with open('saved/training_losses_list.json', 'w') as file:
        json.dump(training_losses, file)

    plt.plot(training_losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss for Two-Stage Faster RCNN\nDetector Using Resnet-50 as Backbone")
    plt.savefig('saved/training_losses.png')
    plt.show()
