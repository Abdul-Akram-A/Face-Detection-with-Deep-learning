import cv2
import tensorflow as tf
import numpy as np
import streamlit as st
import argparse
from PIL import Image
import time
# Streamlit app setup
st.set_page_config(page_title="Real-Time Face Tracking",
                   page_icon="🎥",
                   layout="centered",
                   initial_sidebar_state="collapsed",
                   menu_items={"Get help":"https://github.com/Abdul-Akram-A"})
st.title("Real-time Face Tracking")

# Load your model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("facetracker.h5")
    return model
model=load_model()

# Use session state to manage the recording state
if 'record' not in st.session_state:
    st.session_state.record = False
    st.session_state.stop=False
    
#Start/Stop Button
col1,col2=st.columns([1,1])

with col1:
    if st.button("Record"):
       st.session_state.record=True
       st.session_state.stop=False
with col2:
    if st.button("Stop"):
        st.session_state.record = False
        st.session_state.stop=True

# Placeholder for video frames
frame_placeholder = st.empty()

# Set up argument parser
parser = argparse.ArgumentParser(description='Set camera index for VideoCapture.')
parser.add_argument('--camera', type=int, default=0, help='Camera index')

# Parse arguments
args = parser.parse_args()
camera_index = args.camera

# Function to capture and process video frames
def capture_video():
    cap = cv2.VideoCapture(camera_index)
    while st.session_state.record:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture video frame")
            break
        frame = frame[50:500, 50:500, :]
        color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resize = tf.image.resize(color, (120, 120))
        yhat = model.predict(np.expand_dims(resize / 255.0, 0))
        sample_coords = yhat[1][0]
        
        if yhat[0] > 0.5:
            cv2.rectangle(frame, 
                        tuple(np.multiply(sample_coords[:2], [450, 450]).astype(int)),
                        tuple(np.multiply(sample_coords[2:], [450, 450]).astype(int)), 
                                (255, 0, 0), 2)
            cv2.putText(frame, 'Face Detected', tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                                    [0, -5])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Convert frame to RGB image for display
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_placeholder.image(img)
        
        # Add a short delay to control the frame rate
        time.sleep(0.03)
        
    cap.release()

if st.session_state.record:
    st.success("Face Tracking Activated 🎥")
    capture_video()
elif st.session_state.stop:
    st.warning("Video Tracking is stopped.")

