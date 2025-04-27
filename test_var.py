# test.py
import argparse
import os
import torch
import torchaudio
from model import HybridPercussionSeparator

SAMPLE_RATE = 44100

def load_audio_file(filepath, target_sr=SAMPLE_RATE):
    """
    Loads an arbitrary .wav file and converts it to mono at the target sample rate.
    
    Args:
        filepath (str): Path to the input .wav file.
        target_sr (int): Desired sample rate.
        
    Returns:
        torch.Tensor: Mono waveform tensor of shape (1, num_samples).
    """
    waveform, sr = torchaudio.load(filepath)             # waveform: (channels, num_samples)
    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    # Convert to mono
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform  # (1, num_samples)

def save_audio(filepath, waveform, sr=SAMPLE_RATE):
    torchaudio.save(filepath, waveform, sr)

def separate_track(model, mixture):
    model.eval()
    with torch.no_grad():
        # Add batch dimension: (1, 1, num_samples)
        mixture_batch = mixture.unsqueeze(0)
        estimated_drums = model(mixture_batch)
        estimated_drums = estimated_drums.squeeze(0)
        accompaniment = mixture - estimated_drums
    return estimated_drums, accompaniment

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the pretrained model
    model = HybridPercussionSeparator().to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    # Load the input .wav file
    mixture = load_audio_file(args.input_wav, target_sr=SAMPLE_RATE).to(device)
    
    # Optionally crop to a given duration
    if args.duration is not None:
        num_samples = int(args.duration * SAMPLE_RATE)
        if mixture.size(1) > num_samples:
            mixture = mixture[:, :num_samples]
        else:
            print("Warning: file shorter than specified duration; using full length.")
    
    # Prepare output base name
    base = os.path.splitext(os.path.basename(args.input_wav))[0]
    
    # Save the original mixture
    original_output = f"{base}_original.wav"
    save_audio(original_output, mixture.cpu(), sr=SAMPLE_RATE)
    print(f"Saved original: {original_output}")
    
    # Run separation
    estimated_drums, accompaniment = separate_track(model, mixture)
    
    # Save separated stems
    drums_out = f"{base}_{args.drum_output}"
    acc_out   = f"{base}_{args.accompaniment_output}"
    save_audio(drums_out,      estimated_drums.cpu(), sr=SAMPLE_RATE)
    save_audio(acc_out,        accompaniment.cpu(),    sr=SAMPLE_RATE)
    
    print("Separation complete.")
    print(f"  Drums:         {drums_out}")
    print(f"  Accompaniment: {acc_out}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Separate drums from any input .wav file.")
    parser.add_argument("--input_wav",           type=str, required=True, help="Path to input .wav file.")
    parser.add_argument("--model_path",          type=str, required=True, help="Pretrained model checkpoint.")
    parser.add_argument("--duration",            type=float, default=None, help="Optional: duration (seconds) to crop input to.")
    parser.add_argument("--drum_output",         type=str,   default="drums.wav",          help="Suffix for separated drums file.")
    parser.add_argument("--accompaniment_output",type=str,   default="accompaniment.wav",  help="Suffix for accompaniment file.")
    args = parser.parse_args()
    main(args)
