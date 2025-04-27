import streamlit as st
import torch
import torchaudio
import tempfile
import os
import base64
from model import HybridPercussionSeparator
from streamlit.runtime.scriptrunner import RerunException, RerunData

# Constants
SAMPLE_RATE = 44100
MODEL_CHECKPOINT = "hybrid_percussion_epoch20.pth"  # Fixed checkpoint in same directory
BACKGROUND_IMAGE = "background.png"  # Place your full-page background image here

# Function to set background image
def set_background(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            b64 = base64.b64encode(img_file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{b64}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

# Load and cache the pretrained model
@st.cache_resource
def get_separator():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    separator = HybridPercussionSeparator().to(device)
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location=device)
    separator.load_state_dict(checkpoint)
    separator.eval()
    return separator, device

# Audio processing helper (no caching to avoid hashing issues)
def separate_track(separator, device, mixture, duration=None):
    if duration is not None:
        num_samples = int(duration * SAMPLE_RATE)
        if mixture.size(1) > num_samples:
            mixture = mixture[:, :num_samples]
    with torch.no_grad():
        batch = mixture.unsqueeze(0).to(device)
        drums = separator(batch).squeeze(0)
        accompaniment = mixture.to(device) - drums
    return drums.cpu(), accompaniment.cpu()

# File loader (no caching)
def load_audio_file(filepath, target_sr=SAMPLE_RATE):
    waveform, sr = torchaudio.load(filepath)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform

# Streamlit App
def main():
    # Set full-page background
    set_background(BACKGROUND_IMAGE)

    # Centered and enlarged title
    st.markdown(
        "<h1 style='text-align: center; font-size: 56px; color: white;'>Hybrid Percussion Separation</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center; color: white;'>Upload a WAV file, set a duration, then click Process to separate drums.</p>",
        unsafe_allow_html=True
    )

    # Center inputs using columns
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded = st.file_uploader("Choose a WAV file", type=["wav"], key="uploaded_file")
        duration = st.slider("Max duration (seconds)", min_value=1.0, max_value=60.0, value=10.0, key="duration_slider")
        if uploaded:
            if st.button("Process", key="process_button"):
                separator, device = get_separator()
                with st.spinner("Processing audio..."):
                    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    tmp_in.write(uploaded.read())
                    tmp_in.flush()

                    mixture = load_audio_file(tmp_in.name)
                    drums, acc = separate_track(separator, device, mixture, duration)

                    tmp_drums = tempfile.NamedTemporaryFile(delete=False, suffix="_drums.wav")
                    torchaudio.save(tmp_drums.name, drums, SAMPLE_RATE)
                    tmp_acc = tempfile.NamedTemporaryFile(delete=False, suffix="_acc.wav")
                    torchaudio.save(tmp_acc.name, acc, SAMPLE_RATE)

                # Display results, centered with white text
                st.markdown("<div style='text-align: center; color: white;'><strong>Original Audio</strong></div>", unsafe_allow_html=True)
                st.audio(tmp_in.name, format="audio/wav")
                st.markdown("<div style='text-align: center; color: white;'><strong>Separated Drums</strong></div>", unsafe_allow_html=True)
                st.audio(tmp_drums.name, format="audio/wav")
                st.markdown("<div style='text-align: center; color: white;'><strong>Accompaniment</strong></div>", unsafe_allow_html=True)
                st.audio(tmp_acc.name, format="audio/wav")

            # Reset button
            if st.button("Reset", key="reset_button"):
                if "uploaded_file" in st.session_state:
                    del st.session_state["uploaded_file"]
                raise RerunException(RerunData())

# Entry point
if __name__ == "__main__":
    if not os.path.isfile(MODEL_CHECKPOINT):
        st.error(f"Checkpoint '{MODEL_CHECKPOINT}' not found.")
    else:
        main()
