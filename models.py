import streamlit as st
from ultralytics import YOLO


# Load and cache the person-detection model.
@st.cache_resource
def load_person_model():
    return YOLO("yolov8n.pt")
