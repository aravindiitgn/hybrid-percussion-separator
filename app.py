import os
# Disable Streamlit's file watcher to avoid torch._classes issues on deploy
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import streamlit as st
import tempfile
import base64
import io
from streamlit.runtime.scriptrunner import RerunException, RerunData

# Constants
SAMPLE_RATE = 44100
MODEL_CHECKPOINT = "hybrid_percussion_epoch20.pth"
BACKGROUND_IMAGE = "background.jpg"

# Function to set full-page background image
def set_background(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            b64 = base64.b64encode(img_file.read()).decode()
        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{b64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """, unsafe_allow_html=True)

# Load and cache the pretrained separator model lazily
def get_separator():
    @st.cache_resource
    def _load():
        import torch
        from model import HybridPercussionSeparator
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sep = HybridPercussionSeparator().to(device)
        checkpoint = torch.load(MODEL_CHECKPOINT, map_location=device)
        sep.load_state_dict(checkpoint)
        sep.eval()
        return sep, device
    return _load()

# Load audio file and resample if needed
def load_audio_file(filepath, target_sr=SAMPLE_RATE):
    import torchaudio
    waveform, sr = torchaudio.load(filepath)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform

# Separate drums and accompaniment
def separate_track(separator, device, mixture, duration=None):
    import torch
    if duration is not None:
        num_samples = int(duration * SAMPLE_RATE)
        mixture = mixture[:, :num_samples]
    with torch.no_grad():
        batch = mixture.unsqueeze(0).to(device)
        drums = separator(batch).squeeze(0)
        accompaniment = mixture.to(device) - drums
    return drums.cpu(), accompaniment.cpu()

# Main Streamlit app
def main():
    set_background(BACKGROUND_IMAGE)

    # Semi-transparent grey container behind all content
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] > div:first-child {
            background-color: rgba(128,128,128,0.2) !important;
            padding: 1rem 2rem;
            border-radius: 0.5rem;
        }
        </style>
        """, unsafe_allow_html=True
    )

    # App title and instructions
    st.markdown("<h1 style='text-align:center;font-size:56px;color:white;'>Hybrid Percussion Separation</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:white;'>Upload a WAV file, set the duration, then click Process.</p>", unsafe_allow_html=True)

    # Center the upload & controls
    _, col, _ = st.columns([1, 2, 1])
    with col:
        uploaded = st.file_uploader("Choose a WAV file", type=["wav"])
        duration = st.slider("Max duration (seconds)", min_value=1.0, max_value=60.0, value=10.0)

        if uploaded and st.button("Process"):
            sep, dev = get_separator()
            with st.spinner("Processing audio..."):
                tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                tmp_in.write(uploaded.read()); tmp_in.flush()

                mixture = load_audio_file(tmp_in.name)
                drums, acc = separate_track(sep, dev, mixture, duration)

                t1 = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                t2 = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                import torchaudio
                torchaudio.save(t1.name, drums, SAMPLE_RATE)
                torchaudio.save(t2.name, acc, SAMPLE_RATE)

            for label, path in [("Original Audio", tmp_in.name), ("Separated Drums", t1.name), ("Accompaniment", t2.name)]:
                st.markdown(f"<div style='text-align:center;color:white;'><strong>{label}</strong></div>", unsafe_allow_html=True)
                with open(path, "rb") as f:
                    audio_buffer = io.BytesIO(f.read())
                st.audio(audio_buffer, format="audio/wav")

        if st.button("Reset"):
            raise RerunException(RerunData())

# Entry point
if __name__ == "__main__":
    if not os.path.isfile(MODEL_CHECKPOINT):
        st.error(f"Checkpoint '{MODEL_CHECKPOINT}' not found.")
    else:
        main()
