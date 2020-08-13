# REST API using Flask to interact with a pre-trained Keras model.

[![License](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/AlexJCM/keras-model-with-flask/blob/master/LICENSE)
[![](https://img.shields.io/badge/python-3.5%2B-green.svg)]()
![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)

## Getting started

Requirements (Tensorflow 2.x requires Python 3.5 to 3.7.):

- Python 3.5 to 3.7

Next, clone the repo:

```shell
git clone https://github.com/AlexJCM/keras-model-with-flask.git
cd keras-model-with-flask/
```

Download the following file and put them in the `keras-model-with-flask/resources` folder:

- [keras_weights.h5](https://drive.google.com/file/d/1-6wMqDINf7sK541AQ1ReG5TO_qWAaD9R/view?usp=sharing)

## Install libraries:

Install all the required dependencies with following command:

```shell
pip install -r requirements.txt
```

## Start local server:

```shell
python run_keras_server.py
```

- Go to http://localhost:5000
- Done! :tada:

  Screenshot:

<p align="center">
  <img src="https://i.postimg.cc/xT49NGCg/screenshot-client-side-keras.png" height="180px" alt="">
</p>
