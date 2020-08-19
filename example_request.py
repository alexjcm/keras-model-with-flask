# CLIENT SIDE
# USAGE
# python simple_request.py

import requests

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://localhost:5000/predict"
IMAGE_PATH = "resources/image.jpg"

# load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

# submit the request
r = requests.post(KERAS_REST_API_URL, files=payload).json()  # FIXME

# ensure the request was sucessful
if r["success"]:
    # loop over the predictions and display them
    print(r)
    # for (i, result) in enumerate(r["predictions"]):
    #     print("{}. {}: {:.2f}".format(i + 1, result["label"],
    #                                   result["probability"]))

# otherwise, the request failed
else:
    print("Request failed")
