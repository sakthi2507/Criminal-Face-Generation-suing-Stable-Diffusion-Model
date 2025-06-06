import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
from io import BytesIO
import warnings

warnings.filterwarnings("ignore")

# Page setup
st.set_page_config(page_title="FaceForge: Criminal suspect face Generator", layout="wide")

# Custom CSS for premium look
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
        }

        .main-title {
            font-size: 3.2em;
            font-weight: bold;
            text-align: center;
            margin-bottom: 0.3em;
            color: #112B3C;
        }

        .sub-title {
            text-align: center;
            font-size: 1.1em;
            margin-bottom: 2em;
            color: #555;
        }

        .card {
            background-color: #ffffff;
            padding: 2.5em;
            border-radius: 20px;
            box-shadow: 0px 6px 20px rgba(0, 0, 0, 0.05);
            max-width: 800px;
            margin: auto;
        }

        .footer {
            text-align: center;
            font-size: 0.9em;
            color: #999;
            margin-top: 3em;
        }

        .stTextInput > div > input,
        .stSelectbox > div,
        .stTextArea textarea {
            border-radius: 12px;
        }

        .stButton > button {
            background-color: #164863;
            color: white;
            font-size: 16px;
            padding: 0.5em 1.5em;
            border-radius: 12px;
        }

        .stButton > button:hover {
            background-color: #1E56A0;
        }

        .stDownloadButton button {
            background-color: #28a745;
            color: white;
            border-radius: 12px;
        }

        .stDownloadButton button:hover {
            background-color: #218838;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-title">🕵️ Criminal Face Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-assisted suspect reconstruction based on textual witness reports</div>', unsafe_allow_html=True)

# Sidebar (Case metadata)
with st.sidebar:
    st.header("📁 Case Info")
    st.text_input("🔍 Case ID", "C-9815")
    st.text_input("👮 Officer", "Inspector Leya")
    st.date_input("🗓️ Date")
    st.markdown("---")
    st.caption("🧠 Input detailed suspect description below for best results.")

# Load the model (cached)
@st.cache_resource
def load_pipe():
    pipe = StableDiffusionPipeline.from_pretrained("face_model_finetuned")
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

# Main UI card
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📝 Witness Description")
    age = st.selectbox("Age Group", ["Under 18", "18–30", "31–50", "51+"], index=1)
    gender = st.selectbox("Gender", ["Male", "Female", "Non-binary", "Prefer not to say"], index=0)
    desc = st.text_area("Describe the suspect", 
        "Male, late 30s, sharp jawline, narrow eyes, thin beard, tattoo near the left ear, stern look.")

    generate = st.button("🚀 Reconstruct Face")

    if generate and desc.strip():
        with st.spinner("Generating suspect face..."):
            pipe = load_pipe()
            image = pipe(desc).images[0]

            st.success("✅ Face reconstruction completed.")
            st.image(image, caption="AI-generated face", use_column_width=True)

            # Download
            buf = BytesIO()
            image.save(buf, format="PNG")
            st.download_button("📥 Download Image", buf.getvalue(), file_name="suspect.png", mime="image/png")

            # Placeholder for future matching
            st.info("Face matching system not loaded. Integrate database for results.")

    elif generate:
        st.warning("⚠️ Please enter a description before generating.")

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">🔐 Forensic AI System | Developed for Confidential Use | SRM Research Labs</div>', unsafe_allow_html=True)
