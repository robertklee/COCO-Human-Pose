import imp

import streamlit as st
from streamlit.elements.vega_lite import _CHANNELS

import hourglass

imp.reload(hourglass)
import io
import os
import ssl
import tempfile
import urllib.request

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow import keras

import data_generator
import evaluation
from constants import *
import evaluation_wrapper
from HeatMap import HeatMap
from hourglass import HourglassNet
from util import *
from PIL import Image

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


representative_set_df = pd.read_pickle(os.path.join(DEFAULT_PICKLE_PATH, 'representative_set.pkl'))
subdir = '2021-04-01-21h-59m_batchsize_16_hg_4_loss_weighted_mse_aug_light_sigma4_learningrate_5.0e-03_opt_rmsProp_gt-4kp_activ_sigmoid_subset_0-2.50_lrfix'

def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/robertklee/COCO-Human-Pose/master/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

def run_app_temp(img, f, demo):
    if demo == True:
        rescale_f = cv2.imread(img)
        rescale_f = cv2.cvtColor(rescale_f,cv2.COLOR_BGR2RGB)
        rescale_f = cv2.resize(rescale_f, dsize=(256,256))
        left_column, right_column = st.beta_columns(2)
        left_column.image(rescale_f, caption = "Selected Input")
    else:
        left_column, right_column = st.beta_columns(2)
        left_column.image(f, caption = "Selected Input")

    @st.cache(allow_output_mutation=True, hash_funcs=HASH_FUNCS)
    def load_model(subdir):
        config = tf.ConfigProto(allow_soft_placement=True)
        session = tf.Session(config=config)

        with session.as_default():
            eval = evaluation_wrapper.EvaluationWrapper(modle_sub_dir = subdir, epoch=26)
            eval.visualizeKeypoints(img)


        

    
def run_app(img, f, demo):

    # if demo == True:
    #     rescale_f = cv2.imread(img)
    #     rescale_f = cv2.cvtColor(rescale_f,cv2.COLOR_BGR2RGB)
    #     rescale_f = cv2.resize(rescale_f, dsize=(256,256))
    #     left_column, right_column = st.beta_columns(2)
    #     left_column.image(rescale_f, caption = "Selected Input")
    # else:
    #     left_column, right_column = st.beta_columns(2)
    #     left_column.image(f, caption = "Selected Input")

    left_column, right_column = st.beta_columns(2)
    xb, yb = evaluation.load_and_preprocess_img(img,4)
    left_column.image(xb[0], caption = "Selected Input")


    @st.cache(allow_output_mutation=True, hash_funcs=HASH_FUNCS)
    def load(subdir):
        config = tf.ConfigProto(allow_soft_placement=True)
        session = tf.Session(config=config)
        with session.as_default():
            h = HourglassNet(NUM_COCO_KEYPOINTS,DEFAULT_NUM_HG,INPUT_CHANNELS,INPUT_DIM,OUTPUT_DIM)
            _, val_df = h.load_and_filter_annotations(DEFAULT_TRAIN_ANNOT_PATH,DEFAULT_VAL_ANNOT_PATH,0.1)

            eval = evaluation.Evaluation(
                    model_sub_dir=subdir,
                    epoch=26)

            generator = data_generator.DataGenerator(
                    df=val_df,
                    base_dir=DEFAULT_VAL_IMG_PATH,
                    input_dim=INPUT_DIM,
                    output_dim=OUTPUT_DIM,
                    num_hg_blocks=eval.num_hg_blocks,
                    shuffle=False,  
                    batch_size=len(representative_set_df),
                    online_fetch=False)
            print("Created DataGen Instances")
        return h, val_df, generator, eval, session

    h, val_df, generator, eval, session = load(subdir)
   
    X_batch, y_stacked = evaluation.load_and_preprocess_img(img, eval.num_hg_blocks)
    y_batch = y_stacked[0] # take first hourglass section
    X, y = X_batch[0], y_batch[0] # take first example of batch

    with session.as_default():
        predict_heatmaps=eval.predict_heatmaps(X_batch)
        keypoints = eval.heatmaps_to_keypoints(predict_heatmaps[eval.num_hg_blocks-1, 0, :, :, :]) 


    with session.as_default():
        heatmap = y[:,:,0]
        hm = HeatMap(X,heatmap)
        img_output = np.array(hm.image)
        plt.clf()
        # Plot predicted keypoints on bounding box image
        x = []
        y = []
        for i in range(NUM_COCO_KEYPOINTS):
            if(keypoints[i,0] != 0 and keypoints[i,1] != 0):
                x.append(keypoints[i,0])
                y.append(keypoints[i,1])
        plt.scatter(x,y)
        plt.imshow(img_output)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close('all')
        buf.seek(0)
        right_column.image(buf,  caption = "Predicted Keypoints")
        st.image(buf, caption = 'FINAL: Predicted Pose')
        buf.close()
       
    
def demo():
    left_column, middle_column, right_column = st.beta_columns(3)

    left_column.image('demo_image.jpg', caption = "Demo Image")
    
    middle_column.image('heatmap_result_1.png', caption = "Predicted Heatmap")

    right_column.subheader("Explaination")
    right_column.write("We predict human poses based on key joints.") 

    
def main():

    st.sidebar.write('Please upload SINGLE-pereson images. For best results, please also CENTER the person in the image.')
    st.sidebar.write(" ------ ")
    st.sidebar.title("Explore the Following")
    
    app_mode = st.sidebar.selectbox("Please select from the following",["Show Project Info", "Select a Demo Image", "Upload an Image","Meet the Team"])
    
    if app_mode == "Show Project Info":
        st.sidebar.write(" ------ ")
        st.sidebar.success("Project information showing on the right!")
        st.write(get_file_content_as_string("Project_Info.md"))

    elif app_mode == "Select a Demo Image":
        st.sidebar.write(" ------ ")

        df = pd.DataFrame({'demo images': list(range(1,6))})
        options = st.sidebar.selectbox('Please select a number from 1 to 5, then click Magic Time button', df['demo images'])
        pressed = st.sidebar.button('Magic Time')
        if pressed:
            st.empty()
            st.sidebar.write('Please wait for the magic to happen! It might take up to a few minuates')
            directory = '/Users/wanze/Desktop/SENG_474_Project/COCO-Human-Pose/data/demo_photo'
            photos = [f for f in os.listdir(directory) if(f.endswith('.jpg') or f.endswith('.png')) ]
            pic = os.path.join(directory, photos[options-1])
            run_app(pic, pic, True)

        st.sidebar.write("OR")
        fun = st.sidebar.button('Fur-riend')
        if fun:
           st.sidebar.write('Please enjoy our favourite Kangaroo!') 
           directory = '/Users/wanze/Desktop/SENG_474_Project/COCO-Human-Pose/data/Macropus_rufogriseus_rufogriseus_Bruny.jpg'
           run_app(directory, directory, True)
        #    left_column, right_column = st.beta_columns(2)
        #    kangaroo = cv2.imread(directory)
        #    kangaroo = cv2.cvtColor(kangaroo,cv2.COLOR_BGR2RGB)
        #    kangaroo = cv2.resize(kangaroo, dsize=(256,256))
        #    left_column.image(kangaroo, caption="Hiiii!")
        #    kangaroo_output = cv2.imread('/Users/wanze/Desktop/SENG_474_Project/COCO-Human-Pose/data/Macropus_rufogriseus_rufogriseus_Bruny_saved_scatter.png')
        #    kangaroo_output = cv2.cvtColor(kangaroo_output,cv2.COLOR_BGR2RGB)
        #    kangaroo_output = cv2.resize(kangaroo_output, dsize=(256,256))
        #    right_column.image(kangaroo_output, caption="Have fun!")



    elif app_mode == "Upload an Image":
        #upload = st.empty()
        #with upload:
        st.sidebar.write(" ------ ")
        f = st.sidebar.file_uploader("Please Select to Upload an Image")
        if f is not None:
            tfile = tempfile.NamedTemporaryFile(delete=True)
            tfile.write(f.read())
            run_app(tfile, f, False)


    elif app_mode == "Meet the Team":
        st.sidebar.write(" ------ ")
        st.subheader("We Are the Posers")
        first_column, second_column, third_column, forth_column, fifth_column, sixth_column= st.beta_columns(6)
        
        third_column.image('Wanze.JPG',  caption="Wanze")
        second_column.image('IMG_8175.jpeg', use_column_width = True, caption = "Robert")
        first_column.image(' Julian.jpeg', use_column_width = True, caption = "Julian")
        forth_column.image('IMG_1156-2.jpeg', use_column_width=True, caption = "Nicole")
        fifth_column.image('Rafay.png', use_column_width = True, caption = 'Rafay')
        sixth_column.image('Corey.jpg', use_column_width = True, caption = "Corey")

        first_column_predict, second_column_predict, third_column_predict,forth_column_predict, fifth_column_predict, sixth_column_predict = st.beta_columns(6)
        first_column_predict.image('Julian_output.png', use_column_width = True, caption = "Julian Pose")
        second_column_predict.image('Robert_output.png', use_column_width = True, caption = "Robert Pose")
        third_column_predict.image('Wanze_output.png', use_column_width = True, caption = "Wanze Pose")
        forth_column_predict.image('Nicole_output.png', use_column_width = True, caption = "Nicole Pose")
        fifth_column_predict.image('Rafay_output.png', use_column_width = True, caption = "Rafay Pose")
        sixth_column_predict.image('Corey_output.png', use_column_width = True, caption = "Corey Pose")
       

        st.sidebar.write('Please feel free to connect with us on Linkedin!')
        st.sidebar.success('Hope you had a great time :)')

        expandar_linkedin = st.beta_expander('Contact Information')
        expandar_linkedin.write('Robert: https://www.linkedin.com/in/robert-k-lee/')
        expandar_linkedin.write('Julian: https://www.linkedin.com/in/julianrocha/')
        expandar_linkedin.write('Wanze: https://www.linkedin.com/in/wanze-zhang-59320b137/')
        expandar_linkedin.write('Nicole: https://www.linkedin.com/in/nicole-peverley-64181316a/')
        expandar_linkedin.write('Rafay: https://www.linkedin.com/in/rafay-chaudhy')
        expandar_linkedin.write('Corey: https://www.linkedin.com/in/corey-koelewyn-5b45061ab')
    


main()
expander_faq = st.beta_expander("More About Our Project")
expander_faq.write("Hi there! If you have any questions about our project, or simply want to check out the source code, please visit our github repo: https://github.com/robertklee/COCO-Human-Pose")
