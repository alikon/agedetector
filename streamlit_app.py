# SOURCES:
# https://learnopencv.com/age-gender-classification-using-opencv-deep-learning-c-python/, model loading and usage code taken from there
# https://discuss.streamlit.io/t/remove-made-with-streamlit-from-bottom-of-app/1370/2,
# Hiding the hamburger menu and watermark
import os
import sys
from glob import glob
import time
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes


def get_face_box(net, frame, conf_threshold=0.7):
    opencv_dnn_frame = frame.copy()
    frame_height = opencv_dnn_frame.shape[0]
    frame_width = opencv_dnn_frame.shape[1]
    blob_img = cv2.dnn.blobFromImage(opencv_dnn_frame, 1.0, (300, 300), [
        104, 117, 123], True, False)

    net.setInput(blob_img)
    detections = net.forward()
    b_boxes_detect = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            b_boxes_detect.append([x1, y1, x2, y2])
            cv2.rectangle(opencv_dnn_frame, (x1, y1), (x2, y2),
                          (0, 255, 0), int(round(frame_height / 150)), 8)
    return opencv_dnn_frame, b_boxes_detect

# =======
#   App
# =======
hide_streamlit_style = """
            <style>
            # MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
DEMO_IMAGE = "girl1.jpg"
sys.path.insert(0, ".")
st.write("""
    # Age and Gender prediction
    """)
# provide options to either select an image form the gallery, upload one, or fetch from URL
gallery_tab, upload_tab, url_tab = st.tabs(["Gallery", "Upload", "Image URL"])

with gallery_tab:
    gallery_files = glob(os.path.join(".", "images", "*"))
    gallery_dict = {image_path.split("/")[-1].split(".")[-2].replace("-", " "): image_path
        for image_path in gallery_files}

    options = list(gallery_dict.keys())
    file_name = st.selectbox("Select Art", 
                        options=options, index=options.index("Mona Lisa (Leonardo da Vinci)"))
    file = gallery_dict[file_name]
    if st.session_state.get("file_uploader") is not None:
        st.warning("To use the Gallery, remove the uploaded image first.")
    if st.session_state.get("image_url") not in ["", None]:
        st.warning("To use the Gallery, remove the image URL first.")
    image = Image.open(file)

with upload_tab:
    st.write("## Upload a picture that contains a face")

    uploaded_file = st.file_uploader("Choose a file:")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
   # else:
   #     image = Image.open(DEMO_IMAGE)
with url_tab:
    url_text = st.empty()
    
    # FIXME: the button is a bit buggy, but it's worth fixing this later

    # url_reset = st.button("Clear URL", key="url_reset")
    # if url_reset and "image_url" in st.session_state:
    #     st.session_state["image_url"] = ""
    #     st.write(st.session_state["image_url"])

    url = url_text.text_input("Image URL", key="image_url")
    
    if url!="":
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
        except:
            st.error("The URL does not seem to be valid.")

cap = np.array(image)
cv2.imwrite('temp.jpg', cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY))
cap=cv2.imread('temp.jpg')
face_txt_path="opencv_face_detector.pbtxt"
face_model_path="opencv_face_detector_uint8.pb"
age_txt_path="age_deploy.prototxt"
age_model_path="age_net.caffemodel"
gender_txt_path="gender_deploy.prototxt"
gender_model_path="gender_net.caffemodel"
MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
age_classes=['Age: ~1-2', 'Age: ~3-5', 'Age: ~6-14', 'Age: ~16-22',
               'Age: ~25-30', 'Age: ~32-40', 'Age: ~45-50', 'Age: age is greater than 60']
gender_classes = ['Male', 'Female']
msg=[]

age_net = cv2.dnn.readNet(age_model_path, age_txt_path)
gender_net = cv2.dnn.readNet(gender_model_path, gender_txt_path)
face_net = cv2.dnn.readNet(face_model_path, face_txt_path)
padding = 20
t = time.time()
frameFace, b_boxes = get_face_box(face_net, cap)
if not b_boxes:
    st.write("No face Detected, Checking next frame")
for bbox in b_boxes:
    face = cap[max(0, bbox[1] -
                   padding): min(bbox[3] +
                                padding, cap.shape[0] -
                                1), max(0, bbox[0] -
                                        padding): min(bbox[2] +
                                                      padding, cap.shape[1] -
                                                      1)]
    blob = cv2.dnn.blobFromImage(
        face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    gender_net.setInput(blob)
    gender_pred_list = gender_net.forward()
    gender = gender_classes[gender_pred_list[0].argmax()]

    age_net.setInput(blob)
    age_pred_list = age_net.forward()
    age = age_classes[age_pred_list[0].argmax()]

    label = "{},{}".format(gender, age)
    cv2.putText(
        frameFace,
        label,
        (bbox[0],
         bbox[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,
         255,
         255),
        2,
        cv2.LINE_AA)
    
    msg.append(
        f"Gender : {gender}, confidence = {round(gender_pred_list[0].max() * 100, 2)}%" +
        '\r\n' +
        f"Age : {age}, confidence = {round(age_pred_list[0].max() * 100, 2)}%")

col1, col2 = st.columns(2)
with col1:
    st.header("Original")
    st.image(image)
with col2:
    st.header("Model")
    st.image(frameFace)
with st.expander("See explanation"):
    for mesg in msg:
       # st.write(age_msg)
        st.write(mesg)
"""

"""
#st.write('new1')
#padding=20
#frame = cap
#genderNet = gender_net
#genderList = gender_classes
#ageNet = age_net
#ageList = ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
#
#resultImg,faceBoxes=highlightFace(face_net,frame)
#if not faceBoxes:
#    print("No face detected")
#
#MODEL_MEAN_VALUES3=(78.4263377603, 87.7689143744, 114.895847746)
#for faceBox in faceBoxes:
#    face=frame[max(0,faceBox[1]-padding):
#               min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
#               :min(faceBox[2]+padding, frame.shape[1]-1)]
#    blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES3, swapRB=False)
#    genderNet.setInput(blob)
#    genderPreds=genderNet.forward()
#    gender=genderList[genderPreds[0].argmax()]
#    st.write(f'Gender: {gender} confidence= {round(genderPreds[0].max()*100, 2)}')
#    ageNet.setInput(blob)
#    agePreds=ageNet.forward()
#    age=ageList[agePreds[0].argmax()]
#    st.write(f'Age: {age[1:-1]} years, confidence= {round(agePreds[0].max()*100, 2)}')
#    cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
##    cv2.imshow("Detecting age and gender", resultImg)