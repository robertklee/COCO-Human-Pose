import imghdr
import os
import shutil
import ssl
import tempfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow import keras

import app_helper
from app_helper import AppHelper
from constants import *
from util import *

HASH_FUNCS = {
    tf.Session : id
}

config = tf.compat.v1.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6

session = tf.compat.v1.Session(config=config)

tf.compat.v1.keras.backend.set_session(session)

st.title('Welcome to Posers')
st.write(" ------ ")
ssl._create_default_https_context = ssl._create_unverified_context

IMAGE_DISPLAY_SIZE = (330, 330)
IMAGE_DIR = 'demo_photo'
TEAM_DIR = 'team'

MODEL_WEIGHTS = f'{DEFAULT_MODEL_BASE_DIR}/hpe_epoch107_.hdf5'
MODEL_JSON = f'{DEFAULT_MODEL_BASE_DIR}/hpe_hourglass_stacks_04_.json'

MODEL_WEIGHTS_DEPLOYMENT_URL = 'https://github.com/robertklee/COCO-Human-Pose/releases/download/v0.1-alpha/hpe_epoch107_.hdf5'
MODEL_JSON_DEPLOYMENT_URL = 'https://github.com/robertklee/COCO-Human-Pose/releases/download/v0.1-alpha/hpe_hourglass_stacks_04_.json'

# Constants for sidebar dropdown
SIDEBAR_OPTION_PROJECT_INFO = "Show Project Info"
SIDEBAR_OPTION_DEMO_IMAGE = "Select a Demo Image"
SIDEBAR_OPTION_UPLOAD_IMAGE = "Upload an Image"
SIDEBAR_OPTION_CAMERA = "Take a Picture"
SIDEBAR_OPTION_MEET_TEAM = "Meet the Team"

SIDEBAR_OPTIONS = [SIDEBAR_OPTION_PROJECT_INFO, SIDEBAR_OPTION_DEMO_IMAGE, SIDEBAR_OPTION_UPLOAD_IMAGE, SIDEBAR_OPTION_CAMERA, SIDEBAR_OPTION_MEET_TEAM]

def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/robertklee/COCO-Human-Pose/master/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")


def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        with open(local_filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    return local_filename

# Modified from https://github.com/thoppe/streamlit-skyAR/blob/master/streamlit_app.py
@st.cache
def ensure_model_exists():

    save_dest = Path(DEFAULT_MODEL_BASE_DIR)
    save_dest.mkdir(exist_ok=True)

    f_checkpoint = Path(MODEL_WEIGHTS)

    if not f_checkpoint.exists():
        with st.spinner("Downloading model weights... this may take up to a few minutes. (~150 MB) Please don't interrupt it."):
            download_file(url=MODEL_WEIGHTS_DEPLOYMENT_URL, local_filename=MODEL_WEIGHTS)

    f_architecture = Path(MODEL_JSON)

    if not f_architecture.exists():
        with st.spinner("Downloading model architecture... this may take a few seconds. Please don't interrupt it."):
            download_file(url=MODEL_JSON_DEPLOYMENT_URL, local_filename=MODEL_JSON)

    return AppHelper(model_weights=MODEL_WEIGHTS, model_json=MODEL_JSON)

#         rescale_f = cv2.imread(img)
#         rescale_f = cv2.cvtColor(rescale_f,cv2.COLOR_BGR2RGB)
#         rescale_f = cv2.resize(rescale_f, dsize=(256,256))

@st.cache(allow_output_mutation=True, hash_funcs=HASH_FUNCS)
def load_model():
    config = tf.ConfigProto(allow_soft_placement=True)
    session = tf.Session(config=config)
    with session.as_default():
        handle = ensure_model_exists()

        # Needed to ensure model can be cached and called
        handle.model._make_predict_function()
        handle.model.summary()

    return handle, session


def run_app(img):

    left_column, right_column = st.columns(2)

    xb, yb = app_helper.load_and_preprocess_img(img, num_hg_blocks=1)
    display_image = cv2.resize(np.array(xb[0]), IMAGE_DISPLAY_SIZE,
                        interpolation=cv2.INTER_LINEAR)

    left_column.image(display_image, caption = "Selected Input")

    handle, session = load_model()

    with session.as_default():
        with session.graph.as_default():
            scatter = handle.predict_in_memory(img, visualize_scatter=True, visualize_skeleton=False)
            skeleton = handle.predict_in_memory(img, visualize_scatter=True, visualize_skeleton=True)

            scatter_img = Image.fromarray(scatter)
            skeleton_img = Image.fromarray(skeleton)

            right_column.image(scatter_img,  caption = "Predicted Keypoints")
            st.image(skeleton_img, caption = 'FINAL: Predicted Pose')

def demo():
    left_column, middle_column, right_column = st.columns(3)

    left_column.image(os.path.join(DEFAULT_DATA_BASE_DIR, TEAM_DIR,'skier.png'), caption = "Demo Image")

    middle_column.image(os.path.join(DEFAULT_DATA_BASE_DIR, TEAM_DIR, 'skier_output.png'), caption = "Predicted Heatmap")

    right_column.subheader("Explanation")
    right_column.write("We predict human poses based on key joints.")

def main():

    st.sidebar.warning('\
        Please upload SINGLE-person images. For best results, please also CENTER the person in the image.')
    st.sidebar.write(" ------ ")
    st.sidebar.title("Explore the Following")

    app_mode = st.sidebar.selectbox("Please select from the following", SIDEBAR_OPTIONS)

    if app_mode == SIDEBAR_OPTION_PROJECT_INFO:
        st.sidebar.write(" ------ ")
        st.sidebar.success("Project information showing on the right!")
        st.write(get_file_content_as_string("Project_Info.md"))

    elif app_mode == SIDEBAR_OPTION_DEMO_IMAGE:
        st.sidebar.write(" ------ ")

        directory = os.path.join(DEFAULT_DATA_BASE_DIR, IMAGE_DIR)

        photos = []
        for file in os.listdir(directory):
            filepath = os.path.join(directory, file)

            # Find all valid images
            if imghdr.what(filepath) is not None:
                photos.append(file)

        photos.sort()

        option = st.sidebar.selectbox('Please select a sample image, then click Magic Time button', photos)
        pressed = st.sidebar.button('Magic Time')
        if pressed:
            st.empty()
            st.sidebar.write('Please wait for the magic to happen! This may take up to a minute.')

            pic = os.path.join(directory, option)

            run_app(pic)

        st.sidebar.write("OR")
        fun = st.sidebar.button('Fur-riend')
        if fun:
            st.sidebar.write('Please enjoy our favourite Kangaroo!')
            k = os.path.join(DEFAULT_DATA_BASE_DIR, 'Macropus_rufogriseus_rufogriseus_Bruny.jpg')
            run_app(k)

    elif app_mode == SIDEBAR_OPTION_UPLOAD_IMAGE:
        st.sidebar.info('PRIVACY POLICY: uploaded images are never saved or stored. They are held entirely within memory for prediction \
            and discarded after the final results are displayed. ')
        f = st.sidebar.file_uploader("Please Select to Upload an Image", type=['png', 'jpg', 'jpeg', 'tiff', 'gif'])
        if f is not None:
            tfile = tempfile.NamedTemporaryFile(delete=True)
            tfile.write(f.read())
            st.sidebar.write('Please wait for the magic to happen! This may take up to a minute.')
            run_app(tfile)

    elif app_mode == SIDEBAR_OPTION_CAMERA:
        st.sidebar.info('PRIVACY POLICY: uploaded images are never saved or stored. They are held entirely within memory for prediction \
            and discarded after the final results are displayed.')
        
        img = st.camera_input("Please take a photo with a person in the center of the image. By submitting an image, you agree to the \
            privacy policy.")

        if img is not None:
            tfile = tempfile.NamedTemporaryFile(delete=True)
            tfile.write(img.read())
            st.sidebar.write('Please wait for the magic to happen! This may take up to a minute.')
            run_app(tfile)

    elif app_mode == SIDEBAR_OPTION_MEET_TEAM:
        st.sidebar.write(" ------ ")
        st.subheader("We are the Posers")
        first_column, second_column, third_column, forth_column, fifth_column, sixth_column = st.columns(6)

        third_column.image(os.path.join(DEFAULT_DATA_BASE_DIR, TEAM_DIR, 'wanze.jpg'),      use_column_width = True, caption = "Wanze")
        second_column.image(os.path.join(DEFAULT_DATA_BASE_DIR, TEAM_DIR, 'robert.png'),    use_column_width = True, caption = "Robert")
        first_column.image(os.path.join(DEFAULT_DATA_BASE_DIR, TEAM_DIR, 'julian.jpg'),     use_column_width = True, caption = "Julian")
        forth_column.image(os.path.join(DEFAULT_DATA_BASE_DIR, TEAM_DIR, 'nicole.jpg'),     use_column_width = True, caption = "Nicole")
        fifth_column.image(os.path.join(DEFAULT_DATA_BASE_DIR, TEAM_DIR, 'rafay.png'),      use_column_width = True, caption = 'Rafay')
        sixth_column.image(os.path.join(DEFAULT_DATA_BASE_DIR, TEAM_DIR, 'corey.jpg'),      use_column_width = True, caption = "Corey")

        first_column_predict, second_column_predict, third_column_predict,forth_column_predict, fifth_column_predict, sixth_column_predict = st.columns(6)
        third_column_predict.image(os.path.join(DEFAULT_DATA_BASE_DIR, TEAM_DIR, 'wanze_output.png'),       use_column_width = True, caption = "Wanze Pose")
        second_column_predict.image(os.path.join(DEFAULT_DATA_BASE_DIR, TEAM_DIR, 'robert_output.png'),     use_column_width = True, caption = "Robert Pose")
        first_column_predict.image(os.path.join(DEFAULT_DATA_BASE_DIR, TEAM_DIR, 'julian_output.png'),      use_column_width = True, caption = "Julian Pose")
        forth_column_predict.image(os.path.join(DEFAULT_DATA_BASE_DIR, TEAM_DIR, 'nicole_output.png'),      use_column_width = True, caption = "Nicole Pose")
        fifth_column_predict.image(os.path.join(DEFAULT_DATA_BASE_DIR, TEAM_DIR, 'rafay_output.png'),       use_column_width = True, caption = "Rafay Pose")
        sixth_column_predict.image(os.path.join(DEFAULT_DATA_BASE_DIR, TEAM_DIR, 'corey_output.png'),       use_column_width = True, caption = "Corey Pose")


        st.sidebar.write('Please feel free to connect with us on Linkedin!')
        st.sidebar.success('Hope you had a great time :)')

        expandar_linkedin = st.expander('Contact Information')
        expandar_linkedin.write('Robert: https://www.linkedin.com/in/robert-k-lee/')
        expandar_linkedin.write('Julian: https://www.linkedin.com/in/julianrocha/')
        expandar_linkedin.write('Wanze: https://www.linkedin.com/in/wanze-zhang-59320b137/')
        expandar_linkedin.write('Nicole: https://www.linkedin.com/in/nicole-peverley-64181316a/')
        expandar_linkedin.write('Rafay: https://www.linkedin.com/in/rafay-chaudhy')
        expandar_linkedin.write('Corey: https://www.linkedin.com/in/corey-koelewyn-5b45061ab')
    else:
        raise ValueError('Selected sidebar option is not implemented. Please open an issue on Github: https://github.com/robertklee/COCO-Human-Pose')

main()
expander_faq = st.expander("More About Our Project")
expander_faq.write("Hi there! If you have any questions about our project, or simply want to check out the source code, please visit our github repo: https://github.com/robertklee/COCO-Human-Pose")
