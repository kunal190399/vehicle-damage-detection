import sys, tempfile
from pathlib import Path
import streamlit as st
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from damage_ai.inference import load_model,predict
st.title("Vehicle Damage Detection")
st.warning("Research prototype. Predictions require assessor review.")
weights=st.sidebar.text_input("YOLO checkpoint","runs/detect/damage/weights/best.pt")
conf=st.sidebar.slider("Confidence",.05,.95,.25)
image=st.file_uploader("Upload vehicle image",type=["jpg","jpeg","png"])
if image:
    st.image(image,use_container_width=True)
    if st.button("Detect damage"):
        if not Path(weights).exists():
            st.error("No trained checkpoint. Follow the dataset and training steps in README.")
        else:
            with tempfile.NamedTemporaryFile(suffix=Path(image.name).suffix) as f:
                f.write(image.getvalue()); f.flush()
                st.json(predict(load_model(weights),f.name,conf))
