import os
import streamlit as st
from ultralytics import YOLO

from settings import (
    LOCAL_WEAPON_MODEL,
    WEAPON_REPO,
    WEAPON_FILE,
    WEAPON_CLASSES
)


@st.cache_resource
def load_person_model():
    return YOLO("yolov8n.pt")


@st.cache_resource
def load_weapon_model():
    if os.path.exists(LOCAL_WEAPON_MODEL):
        model_path = LOCAL_WEAPON_MODEL
    else:
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(
            repo_id=WEAPON_REPO,
            filename=WEAPON_FILE
        )

    model = YOLO(model_path)

    weapon_ids = [
        class_id
        for class_id, class_name in model.names.items()
        if str(class_name).strip().lower() in WEAPON_CLASSES
    ]

    if not weapon_ids:
        weapon_ids = [
            class_id
            for class_id, class_name in model.names.items()
            if str(class_name).strip().lower() != "explosive"
        ]

    return model, weapon_ids