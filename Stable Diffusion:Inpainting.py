import cv2
import numpy as np
import os
from PIL import Image
import streamlit as st
import tempfile
import io
import base64
from scripts.inpaint import inpaint
import shutil

def get_image_download_link(img,filename,text):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href =  f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href

def resize(orig_img_path, orig_mask_path, img_path, mask_path):
  im = cv2.imread(orig_img_path)
  mask = cv2.imread(orig_mask_path,0)
  aspect_ratio = im.shape[0]/im.shape[1]
  if im.shape[0] > im.shape[1]:
    w = (int(512 / aspect_ratio))//64
    size = (64*w,512)
  else:
    h = (int(512* aspect_ratio))//64
    size = (512, 64*h)
  cv2.imwrite(img_path, cv2.resize(im, size))
  cv2.imwrite(mask_path, cv2.resize(mask, size))

def get_detections(orig_img_path, det_path):
    from detect_objects import Detectobjects
    detect = Detectobjects(orig_img_path)
    detect.save_predictions(det_path)
    masks_indexes = [i for i in range(len(detect.masks))]

    del detect
    return masks_indexes

def fetch_prediction(orig_img_path, orig_mask_path, mask_indexes):
    from detect_objects import Detectobjects, resize
    detect = Detectobjects(orig_img_path)
    detect.fetch_predictions(mask_indexes, orig_mask_path)
    del detect

def main():
    with tempfile.TemporaryDirectory() as tmpdirname:
        st.markdown("<h4 style='text-align: center;'>Are your moments photobomed? Try this</h1>", unsafe_allow_html=True)
        st.image('inpainting-ui.png', use_column_width=True)

        st.markdown("<h1 style='text-align: center;'>Edit your Image using Stable Diffusion</h1>", unsafe_allow_html=True)
        image_file = st.file_uploader("Upload Image",
                                        type=["png", "jpg", "jpeg"])                      
        if image_file is not None:
            with open(os.path.join(tmpdirname, image_file.name.split('.')[0] +'.png') , "wb") as f:
                f.write(image_file.getbuffer())
            orig_img_path = os.path.join(tmpdirname, image_file.name.split('.')[0] +'.png')
            org_size = Image.open(orig_img_path).size
            det_path = os.path.join(tmpdirname, image_file.name.split('.')[0] +'-detect.png')
            masks_indexes = get_detections(orig_img_path, det_path)
        
            images = [orig_img_path,det_path]
            cols = st.columns(2)
            cols[0].image(images[0], use_column_width=True, caption=['Your Image'])
            cols[1].image(images[1], use_column_width=True, caption=['Masks Detected'])

            st.markdown("<h3 style='text-align: center;'>Choose the Persons/Objects who/which you want to keep along with you</h3>", unsafe_allow_html=True)
            with st.form("Select your options"):
                options = st.multiselect(
                            'Select your options',
                            masks_indexes)
                button = st.form_submit_button(label='Do the magic')

            if button:
                if len(options) > 0:
                    print(options)
                    orig_mask_path =os.path.join(tmpdirname, image_file.name.split('.')[0] +'_mask.png')
                    fetch_prediction(orig_img_path, orig_mask_path, options)
                    os.makedirs(os.path.join(tmpdirname,'mask'), exist_ok=True)
                    
                    img_path = os.path.join(tmpdirname,'mask',image_file.name.split('.')[0] +'.png')
                    mask_path = os.path.join(tmpdirname,'mask',image_file.name.split('.')[0] +'_mask.png')
                    resize(orig_img_path, orig_mask_path, img_path, mask_path)

                    inpainted = inpaint(os.path.join(tmpdirname,'mask'))
                    # output_image = show_image("/Users/ramji/Projects/personal-projects/inpaint_output.png")
                    # st.image(inpainted, use_column_width=True)
                    inpainted = cv2.resize(inpainted, org_size)
                    inpainted = Image.fromarray(inpainted)
                    print(org_size, inpainted.size)
                    inpainted.save(os.path.join(tmpdirname,'inpainted.jpg'))

                    images = [orig_img_path,os.path.join(tmpdirname,'inpainted.jpg')]
                    cols = st.columns(2)
                    cols[0].image(images[0], use_column_width=True, caption=['Your Image'])
                    cols[1].image(images[1], use_column_width=True, caption=['Inpainted Image'])

                    st.markdown(get_image_download_link(inpainted,image_file.name,'Download'), unsafe_allow_html=True)
                else:
                    print("Select atleast one mask")
            else:
                print("Select atleast one mask")

if __name__ == '__main__':
    main()