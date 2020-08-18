# SERVER SIDE
# USAGE
# Start the server:
# 	python main.py
# Submit a request via Python:
# python example_request.py
import io
import flask
from tensorflow import keras
import numpy as np
from PIL import Image
from keras.models import model_from_json
from keras.models import load_model
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array, load_img

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None


# Para cargar nuestro modelo Keras entrenado y prepararlo para la inferencia
def load_my_model():
    global model
    # model = load_model('keras_model.h5')  # v3
    with open('resources/keras_model.json') as json_file:
        json_config = json_file.read()
    model = model_from_json(json_config)
    model.load_weights('resources/keras_weights.h5')

# Procesa previamente una imagen de entrada antes de pasarla a través
# de nuestra red para la predicción.


def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    # image = imagenet_utils.preprocess_input(image)

    return image  # return the processed image


def decode_predictionss(preds, top, class_list_path):
    if len(preds.shape) != 2 or preds.shape[1] != 2:  # your classes number
        raise ValueError('`XXXXXXXXXXXX decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 1000)). '
                         'Found array with shape: ' + str(preds.shape))

    index_list = flask.json.load(open(class_list_path))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(index_list[str(i)]) + (pred[i],) for i in top_indices]
        results.append(result)
    return results


# El endpoint real de nuestra API que clasificará los datos entrantes de la
# solicitud y devolverá los resultados al cliente.
@app.route("/predict", methods=["POST"])
def predict():
        # initialize the data dictionary that will be returned from the
        # view
    data = {"success": False}

    # Asegúrate de que una imagen se suba correctamente a nuestro endpoint.
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocesar la imagen y prepararla para su clasificación.
            # Se pone en target el mismo tamaño establecido durante el entrenamiento.
            image = prepare_image(image, target=(224, 224))

            # Clasificar la imagen de entrada y luego inicializar la lista
            # de predicciones para retornarla al cliente
            preds = model.predict(image)
            results = decode_predictionss(
                preds, 4, 'resources/labels.json')

            data["predictions"] = []

            # loop over the results and add them to the list of
            # returned predictions
            for (label, prob) in results[0]:
                r = {"label": label, "probability": float(prob)}
                data["predictions"].append(r)

            # ************EXTRAAAA********************
            result = preds[0]
            # arroja la posicion donde esta el valor mayor
            answer = np.argmax(result)
            if answer == 0:
                my_label = "benign"
            elif answer == 1:
                my_label = "malignant"
            r2 = {"my_label": my_label, "probability": float(answer)}
            data["predictions"].append(r2)
            # ********************************

            # Indique que la solicitud fue un éxito.
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


@app.route("/")
def index():
    return "Welcome to the client side of our Keras model. :)"


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_my_model()
    app.run(port=5000)
