import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from dataset import MUSDB18Dataset, SAMPLE_RATE
from model import HybridPercussionSeparator

def spectral_loss(y_true, y_pred, n_fft=1024, hop_length=256):
    window = torch.hann_window(n_fft).to(y_true.device)
    Y  = torch.stft(y_true.squeeze(1), n_fft=n_fft, hop_length=hop_length,
                    window=window, return_complex=True)
    Yp = torch.stft(y_pred.squeeze(1), n_fft=n_fft, hop_length=hop_length,
                    window=window, return_complex=True)
    mag  = torch.abs(Y)
    magp = torch.abs(Yp)
    sc_loss = torch.norm(magp - mag, p='fro') / (torch.norm(mag, p='fro') + 1e-8)
    log_l1  = F.l1_loss(torch.log(mag + 1e-7), torch.log(magp + 1e-7))
    return sc_loss + log_l1

def train_model():
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size  = 4
    num_epochs  = 90
    lr          = 1e-4
    lambda_spec = 0.05   # lowered from 0.1 to reduce muffling

    # dataset & loader
    dataset = MUSDB18Dataset(subset="train", duration=4.0, download=True)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    total_steps = len(loader)

    # model, optimizer, scheduler, criteria
    model        = HybridPercussionSeparator().to(device)
    optimizer    = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler    = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion_l1 = nn.L1Loss()

    print(f"⏳ Training on {len(dataset)} segments ({total_steps} batches/epoch)")
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(enumerate(loader, 1), total=total_steps, desc=f"Epoch {epoch}/{num_epochs}")

        for batch_idx, (mixture, drums) in pbar:
            mixture, drums = mixture.to(device), drums.to(device)
            optimizer.zero_grad()

            est_drums = model(mixture)
            loss_l1   = criterion_l1(est_drums, drums)
            loss_spec = spectral_loss(drums, est_drums)
            loss      = loss_l1 + lambda_spec * loss_spec

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(batch_loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / total_steps
        print(f"→ Epoch {epoch:>2} complete. Avg Loss: {avg_loss:.4f}")
        scheduler.step(avg_loss)

        if epoch % 10 == 0:
            ckpt = f"hybrid_percussion_epoch{epoch}.pth"
            torch.save(model.state_dict(), ckpt)
            print(f"💾 Saved checkpoint: {ckpt}")

    print("✅ Training finished.")

if __name__ == "__main__":
    train_model()
