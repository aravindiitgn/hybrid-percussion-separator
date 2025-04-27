import argparse
import torch
import torchaudio
import musdb
from model import HybridPercussionSeparator

SAMPLE_RATE = 44100

def load_musdb_track(track_idx, subset="test", target_sr=SAMPLE_RATE):
    """
    Loads a track from the MUSDB18 dataset.
    
    Args:
        track_idx (int): Index of the track in the specified subset.
        subset (str): 'train' or 'test' subset.
        target_sr (int): Target sample rate.
        
    Returns:
        torch.Tensor: Mixture waveform as a tensor of shape (1, num_samples).
    """
    db = musdb.DB(subsets=[subset], download=True)
    track = db[track_idx]
    mixture = track.audio  # shape: (nb_samples, 2)
    
    # Resample if needed.
    if track.rate != target_sr:
        mixture_tensor = torch.tensor(mixture.T, dtype=torch.float32)
        resampler = torchaudio.transforms.Resample(orig_freq=track.rate, new_freq=target_sr)
        mixture = resampler(mixture_tensor).numpy().T

    # Convert to mono by averaging channels if stereo.
    if mixture.ndim == 2 and mixture.shape[1] > 1:
        mixture = mixture.mean(axis=1, keepdims=True)
        
    # Convert to torch tensor with shape (1, num_samples)
    mixture = torch.tensor(mixture.T, dtype=torch.float32)
    return mixture

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
    
    # Load the pretrained model.
    model = HybridPercussionSeparator().to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    # Load a track from MUSDB18 using the specified subset and track index.
    mixture = load_musdb_track(args.track_idx, subset=args.subset, target_sr=SAMPLE_RATE).to(device)
    
    # If a duration is specified, crop the track to that duration.
    if args.duration is not None:
        num_samples = int(args.duration * SAMPLE_RATE)
        if mixture.size(1) > num_samples:
            mixture = mixture[:, :num_samples]
        else:
            print("Warning: Track is shorter than the specified duration. Using full track.")
    
    # Save the original mixture before applying separation.
    original_output = f"track{args.track_idx}_original.wav"
    save_audio(original_output, mixture.cpu(), sr=SAMPLE_RATE)
    print(f"Original track saved to: {original_output}")
    
    # Run separation.
    estimated_drums, accompaniment = separate_track(model, mixture)
    
    # Append track_idx to the output file names.
    drum_output = f"track{args.track_idx}_{args.drum_output}"
    accompaniment_output = f"track{args.track_idx}_{args.accompaniment_output}"
    
    # Save separated outputs.
    save_audio(drum_output, estimated_drums.cpu(), sr=SAMPLE_RATE)
    save_audio(accompaniment_output, accompaniment.cpu(), sr=SAMPLE_RATE)
    
    print("Separation complete.")
    print(f"Drums track saved to: {drum_output}")
    print(f"Accompaniment track saved to: {accompaniment_output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Separate drums from a MUSDB18 track.")
    parser.add_argument("--track_idx", type=int, required=True, help="Index of the track in MUSDB18.")
    parser.add_argument("--subset", type=str, default="test", help="MUSDB18 subset to use ('train' or 'test').")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained model checkpoint.")
    parser.add_argument("--duration", type=float, default=None, help="Duration (in seconds) to process from the track. If not provided, the full track is used.")
    parser.add_argument("--drum_output", type=str, default="drums.wav", help="Output path for separated drums.")
    parser.add_argument("--accompaniment_output", type=str, default="accompaniment.wav", help="Output path for accompaniment.")
    args = parser.parse_args()
    main(args)
