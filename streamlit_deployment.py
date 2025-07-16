import streamlit as st
import cv2
import numpy as np
import tempfile
import tensorflow as tf

# Load your trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("LRCN_model.h5")  # Change to your model file
    return model

model = load_model()

# Define input dimensions
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64  # Update based on your model input size
SEQUENCE_LENGTH = 20  # Adjust based on your model
CLASSES_LIST = ["Violence", "NonViolence"]

# Function to predict violence in a video
def predict_on_video(video_file_path, SEQUENCE_LENGTH):
    video_reader = cv2.VideoCapture(video_file_path)
    frames_list = []
    
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)
    
    for frame_counter in range(SEQUENCE_LENGTH):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        if not success:
            break
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255.0
        frames_list.append(normalized_frame)
    
    video_reader.release()
    
    if len(frames_list) < SEQUENCE_LENGTH:
        return "Error: Not enough frames in the video!"
    
    predicted_labels_probabilities = model.predict(np.expand_dims(frames_list, axis=0))[0]
    predicted_label = np.argmax(predicted_labels_probabilities)
    predicted_class_name = CLASSES_LIST[predicted_label]
    return f'🚨 Action Predicted: {predicted_class_name}'

# Streamlit UI
st.markdown("<h1 style='text-align: center; color: white;'>🎥 Violence Detection in Videos</h1>", unsafe_allow_html=True)

st.markdown("---")
st.write("Upload a video to check if it contains violence.")

uploaded_file = st.file_uploader("📂 Upload Video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name
    
    st.video(video_path)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Detect Violence", use_container_width=True):
            with st.spinner("Analyzing video... Please wait..."):
                result = predict_on_video(video_path, SEQUENCE_LENGTH)
            st.success(result)

st.markdown("---")
st.markdown("<h4 style='text-align: center; color: gray;'>Made by Deep Learning</h4>", unsafe_allow_html=True)
