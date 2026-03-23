import io
import logging
import os
import shutil
import ssl
import tempfile
import time
import traceback
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

import pillow_heif
pillow_heif.register_heif_opener()

import app_helper
from app_helper import AppHelper
from constants import *
from util import *

# Configure GPU memory growth for TF2
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

st.title('Welcome to Posers')
st.write(" ------ ")
ssl._create_default_https_context = ssl._create_unverified_context

IMAGE_DISPLAY_SIZE = (330, 330)
IMAGE_DIR = 'demo_photo'
TEAM_DIR = 'team'

MODEL_WEIGHTS_KERAS = f'{DEFAULT_MODEL_BASE_DIR}/hpe_epoch107_.keras'
MODEL_WEIGHTS_HDF5 = f'{DEFAULT_MODEL_BASE_DIR}/hpe_epoch107_.hdf5'
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
@st.cache_resource(show_spinner="Loading model... this may take a moment on first run.")
def load_model():
    logging.info("Loading model weights and architecture...")

    save_dest = Path(DEFAULT_MODEL_BASE_DIR)
    save_dest.mkdir(exist_ok=True)

    # Prefer .keras format if available, otherwise fall back to .hdf5
    f_keras = Path(MODEL_WEIGHTS_KERAS)
    f_hdf5 = Path(MODEL_WEIGHTS_HDF5)

    if f_keras.exists():
        model_weights = MODEL_WEIGHTS_KERAS
    elif f_hdf5.exists():
        model_weights = MODEL_WEIGHTS_HDF5
    else:
        with st.spinner("Downloading model weights... this may take up to a few minutes. (~150 MB) Please don't interrupt it."):
            download_file(url=MODEL_WEIGHTS_DEPLOYMENT_URL, local_filename=MODEL_WEIGHTS_HDF5)
        model_weights = MODEL_WEIGHTS_HDF5

    f_architecture = Path(MODEL_JSON)

    if not f_architecture.exists():
        with st.spinner("Downloading model architecture... this may take a few seconds. Please don't interrupt it."):
            download_file(url=MODEL_JSON_DEPLOYMENT_URL, local_filename=MODEL_JSON)

    return AppHelper(model_weights=model_weights, model_json=MODEL_JSON)

PREDICTION_CACHE_PREFIX = "prediction_"
TEAM_CACHE_KEY = "team_predictions"

def _clear_prediction_cache():
    """Remove all cached prediction data from session state."""
    keys_to_remove = [k for k in st.session_state
                      if k.startswith(PREDICTION_CACHE_PREFIX) or k == TEAM_CACHE_KEY]
    for k in keys_to_remove:
        del st.session_state[k]

def run_app(img):

    handle = load_model()

    left_column, right_column = st.columns(2)

    # Display the original image immediately so the page isn't blank during inference.
    # Preview is shown in a small column, so cap to a reasonable size.
    with Image.open(img) as orig_img:
        from PIL import ImageOps
        orig_img = ImageOps.exif_transpose(orig_img.convert('RGB'))
        w, h = orig_img.size
        if max(w, h) > app_helper.MAX_DIM_SCATTER:
            scale = app_helper.MAX_DIM_SCATTER / max(w, h)
            orig_img = orig_img.resize(
                (int(w * scale), int(h * scale)), Image.LANCZOS)
        display_image = np.array(orig_img)

    left_column.image(display_image, caption="Original Image")

    with right_column, st.spinner("Running pose estimation..."):
        try:
            orig_batch, keypoints_batch, heatmaps, crop_info = handle.predict_in_memory_fullres(img)
        except ValueError as e:
            st.error(str(e))
            return

        # Skeleton: full resolution — the hero output
        skeleton = handle.visualize_keypoints(orig_batch, keypoints_batch, show_skeleton=True)

        # Scatter (keypoints only): capped to MAX_DIM_SCATTER
        scat_batch, scat_kp, _scat_ci = app_helper.downscale_for_display(
            orig_batch, keypoints_batch, crop_info, app_helper.MAX_DIM_SCATTER)
        scatter = handle.visualize_keypoints(scat_batch, scat_kp, show_skeleton=False)
        del scat_batch, scat_kp  # free before heatmap rendering

        # Heatmap overlays: capped to MAX_DIM_HEATMAP (17 images, keep memory bounded)
        hm_batch, _hm_kp, hm_ci = app_helper.downscale_for_display(
            orig_batch, keypoints_batch, crop_info, app_helper.MAX_DIM_HEATMAP)
        heatmap_overlays = {}
        for joint_idx in range(NUM_COCO_KEYPOINTS):
            heatmap_overlays[joint_idx] = handle.visualize_heatmap(
                hm_batch[0], heatmaps[0, :, :, joint_idx], joint_idx,
                crop_info=hm_ci)
        del hm_batch, orig_batch  # free full-res arrays

    right_column.image(scatter, caption = "Predicted Keypoints")

    # Bypass Streamlit's internal MAXIMUM_CONTENT_WIDTH (1460 px) downscaling
    # by passing the actual image width.  The frontend CSS still caps the
    # display size to the container, but the full-res data is preserved so
    # "Save Image" in the browser yields the original resolution.
    st.image(skeleton, caption='Predicted Pose', width=int(skeleton.shape[1]))

    _buf = io.BytesIO()
    Image.fromarray(skeleton).save(_buf, format='JPEG', quality=95)
    st.download_button(
        "⬇ Download Full-Resolution Pose",
        data=_buf.getvalue(),
        file_name="predicted_pose.jpg",
        mime="image/jpeg",
    )

    # Per-joint heatmap visualization grid
    with st.expander("🔥 View Per-Joint Heatmaps"):
        # Center: nose
        for joint_idx, label in HEATMAP_DISPLAY_ORDER_CENTER:
            st.image(heatmap_overlays[joint_idx], caption=label, width=300)

        # Left and right side-by-side, top to bottom
        for (l_idx, l_label), (r_idx, r_label) in zip(
                HEATMAP_DISPLAY_ORDER_LEFT, HEATMAP_DISPLAY_ORDER_RIGHT):
            col_l, col_r = st.columns(2)
            col_l.image(heatmap_overlays[l_idx], caption=l_label, use_container_width=True)
            col_r.image(heatmap_overlays[r_idx], caption=r_label, use_container_width=True)

def demo():
    left_column, middle_column, right_column = st.columns(3)

    left_column.image(os.path.join(DEFAULT_DATA_BASE_DIR, TEAM_DIR,'skier.png'), caption = "Demo Image")

    middle_column.image(os.path.join(DEFAULT_DATA_BASE_DIR, TEAM_DIR, 'skier_output.png'), caption = "Predicted Heatmap")

    right_column.subheader("Explanation")
    right_column.write("We predict human poses based on key joints.")

def main():

    # Eagerly download & load the model on first page visit so it's ready
    # by the time the user navigates to a prediction page.
    load_model()

    st.sidebar.warning('\
        Please upload SINGLE-person images. For best results, please also CENTER the person in the image.')
    st.sidebar.write(" ------ ")
    st.sidebar.title("Explore the Following")

    app_mode = st.sidebar.selectbox("Please select from the following", SIDEBAR_OPTIONS)

    # Clear cached prediction data when navigating away from prediction pages
    prediction_pages = {SIDEBAR_OPTION_DEMO_IMAGE, SIDEBAR_OPTION_UPLOAD_IMAGE, SIDEBAR_OPTION_CAMERA, SIDEBAR_OPTION_MEET_TEAM}
    if app_mode not in prediction_pages:
        _clear_prediction_cache()

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
            try:
                with Image.open(filepath) as test_img:
                    test_img.verify()
                photos.append(file)
            except Exception:
                pass

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
        f = st.sidebar.file_uploader("Please Select to Upload an Image", type=['png', 'jpg', 'jpeg', 'tiff', 'gif', 'webp', 'heic', 'heif'])
        if f is not None:
            suffix = os.path.splitext(f.name)[1] or '.jpg'
            f.seek(0)
            with tempfile.NamedTemporaryFile(delete=False, delete_on_close=False, suffix=suffix) as tfile:
                tfile.write(f.read())
            try:
                st.sidebar.write('Please wait for the magic to happen! This may take up to a minute.')
                run_app(tfile.name)
            finally:
                os.unlink(tfile.name)

    elif app_mode == SIDEBAR_OPTION_CAMERA:
        st.sidebar.info('PRIVACY POLICY: uploaded images are never saved or stored. They are held entirely within memory for prediction \
            and discarded after the final results are displayed.')
        
        img = st.camera_input("Please take a photo with a person in the center of the image. By submitting an image, you agree to the \
            privacy policy.")

        if img is not None:
            img.seek(0)
            with tempfile.NamedTemporaryFile(delete=False, delete_on_close=False, suffix='.jpg') as tfile:
                tfile.write(img.read())
            try:
                st.sidebar.write('Please wait for the magic to happen! This may take up to a minute.')
                run_app(tfile.name)
            finally:
                os.unlink(tfile.name)

    elif app_mode == SIDEBAR_OPTION_MEET_TEAM:
        st.sidebar.write(" ------ ")
        st.subheader("We are the Posers")

        team_members = [
            ('julian',  'julian.jpg',  "Julian"),
            ('robert',  'robert.png',  "Robert"),
            ('wanze',   'wanze.jpg',   "Wanze"),
            ('nicole',  'nicole.jpg',  "Nicole"),
            ('rafay',   'rafay.png',   "Rafay"),
            ('corey',   'corey.jpg',   "Corey"),
        ]

        # Row 1: Display original team photos immediately
        cols_orig = st.columns(len(team_members))
        for col, (_member_id, filename, display_name) in zip(cols_orig, team_members):
            img_path = os.path.join(DEFAULT_DATA_BASE_DIR, TEAM_DIR, filename)
            with Image.open(img_path) as orig_img:
                from PIL import ImageOps
                disp = np.array(ImageOps.exif_transpose(orig_img.convert('RGB')))
            col.image(disp, width="100%", caption=display_name)

        handle = load_model()

        # Run predictions (cached so page doesn't re-run on every rerun)
        if TEAM_CACHE_KEY not in st.session_state:
            with st.spinner("Running pose estimation on team photos..."):
                results = {}
                for member_id, filename, display_name in team_members:
                    img_path = os.path.join(DEFAULT_DATA_BASE_DIR, TEAM_DIR, filename)
                    orig_batch, keypoints_batch, _heatmaps, _crop_info = handle.predict_in_memory_fullres(img_path)
                    skeleton = handle.visualize_keypoints(orig_batch, keypoints_batch, show_skeleton=True)
                    results[member_id] = {
                        'skeleton': skeleton,
                        'display_name': display_name,
                    }
                st.session_state[TEAM_CACHE_KEY] = results

        team_results = st.session_state[TEAM_CACHE_KEY]

        # Row 2: Predicted poses
        cols_pred = st.columns(len(team_members))
        for col, (member_id, _filename, _name) in zip(cols_pred, team_members):
            r = team_results[member_id]
            col.image(r['skeleton'], width="100%", caption=f"{r['display_name']} Pose")


        st.sidebar.write('Please feel free to connect with us on Linkedin!')
        st.sidebar.success('Hope you had a great time :)')

        expandar_linkedin = st.expander('Contact Information')
        expandar_linkedin.write('Robert: https://www.linkedin.com/in/robert-k-lee/')
        expandar_linkedin.write('Julian: https://www.linkedin.com/in/julianrocha/')
        expandar_linkedin.write('Wanze: https://www.linkedin.com/in/wanze-zhang/')
        expandar_linkedin.write('Nicole: https://www.linkedin.com/in/nicole-peverley-64181316a/')
        expandar_linkedin.write('Rafay: https://www.linkedin.com/in/rafay-chaudhy')
        expandar_linkedin.write('Corey: https://www.linkedin.com/in/corey-koelewyn-5b45061ab')
    else:
        raise ValueError('Selected sidebar option is not implemented. Please open an issue on Github: https://github.com/robertklee/COCO-Human-Pose')

MAX_AUTO_RESTARTS = 3
AUTO_RESTART_COOLDOWN = 2  # seconds

if "restart_count" not in st.session_state:
    st.session_state["restart_count"] = 0

try:
    main()
except Exception:
    restart_count = st.session_state["restart_count"]
    logging.error("Unexpected error (restart %d/%d):\n%s",
                  restart_count + 1, MAX_AUTO_RESTARTS, traceback.format_exc())

    if restart_count < MAX_AUTO_RESTARTS:
        st.session_state["restart_count"] = restart_count + 1
        st.error(f"Something went wrong. Restarting automatically "
                 f"({restart_count + 1}/{MAX_AUTO_RESTARTS})...")
        time.sleep(AUTO_RESTART_COOLDOWN)
        st.rerun()
    else:
        st.session_state["restart_count"] = 0
        st.error("The app encountered repeated errors and could not recover. "
                 "Please refresh the page to try again.")
        st.exception(Exception(traceback.format_exc()))
else:
    # Successful run — reset the restart counter.
    st.session_state["restart_count"] = 0

expander_faq = st.expander("More About Our Project")
expander_faq.write("Hi there! If you have any questions about our project, or simply want to check out the source code, please visit our github repo: https://github.com/robertklee/COCO-Human-Pose")
