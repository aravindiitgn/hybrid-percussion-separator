import torch
import torch.nn as nn
import torch.nn.functional as F

# ───────────────────────────────────────────────────────────────────────────────
# Residual 1D Block with BatchNorm
# ───────────────────────────────────────────────────────────────────────────────
class Res1DBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn2   = nn.BatchNorm1d(channels)
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(residual + out)

# ───────────────────────────────────────────────────────────────────────────────
# Squeeze-Excitation Block
# ───────────────────────────────────────────────────────────────────────────────
class SqueezeExcite2D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1  = nn.Conv2d(channels, channels // reduction, 1)
        self.fc2  = nn.Conv2d(channels // reduction, channels, 1)
    def forward(self, x):
        w = self.pool(x)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w

# ───────────────────────────────────────────────────────────────────────────────
# Time-Domain Branch with ResBlocks + BatchNorm
# ───────────────────────────────────────────────────────────────────────────────
class TimeDomainBranch(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )
        # stack of residual blocks at increasing dilation
        self.resblocks = nn.ModuleList([
            Res1DBlock(channels, kernel_size=3, dilation=d)
            for d in (1, 2, 4, 8)
        ])
        self.final = nn.Conv1d(channels, 1, kernel_size=15, padding=7)

    def forward(self, x):
        x = self.initial(x)
        for blk in self.resblocks:
            x = blk(x)
        return self.final(x)  # (B,1,T)

# ───────────────────────────────────────────────────────────────────────────────
# Complex Mask UNet with extra depth + SE blocks
# ───────────────────────────────────────────────────────────────────────────────
class ComplexMaskUNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=16):
        super().__init__()
        # 4-level encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
            SqueezeExcite2D(base_channels)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*2), nn.ReLU(inplace=True),
            SqueezeExcite2D(base_channels*2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*4), nn.ReLU(inplace=True),
            SqueezeExcite2D(base_channels*4)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*8, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*8), nn.ReLU(inplace=True),
            SqueezeExcite2D(base_channels*8)
        )

        # 4-level decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*8, base_channels*4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*4), nn.ReLU(inplace=True),
            SqueezeExcite2D(base_channels*4)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*8, base_channels*2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*2), nn.ReLU(inplace=True),
            SqueezeExcite2D(base_channels*2)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*4, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
            SqueezeExcite2D(base_channels)
        )

        # final mask conv (real & imag)
        self.final_conv = nn.Conv2d(base_channels*2, 2, kernel_size=3, padding=1)

    def crop(self, src, tgt):
        _,_,h,w = src.shape
        _,_,ht,wt = tgt.shape
        sh, sw = (h-ht)//2, (w-wt)//2
        return src[:,:,sh:sh+ht, sw:sw+wt]

    def forward(self, mag):
        e1 = self.enc1(mag)       # (B,C, F, T)
        e2 = self.enc2(e1)        # (B,2C, F/2, T/2)
        e3 = self.enc3(e2)        # (B,4C, F/4, T/4)
        e4 = self.enc4(e3)        # (B,8C, F/8, T/8)

        d3 = self.dec3(e4)        # (B,4C, F/4, T/4)
        if d3.shape[2:] != e3.shape[2:]:
            d3 = self.crop(d3, e3)
        d3 = torch.cat([d3, e3], dim=1)

        d2 = self.dec2(d3)        # (B,2C, F/2, T/2)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = self.crop(d2, e2)
        d2 = torch.cat([d2, e2], dim=1)

        d1 = self.dec1(d2)        # (B,  C, F, T)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = self.crop(d1, e1)
        d1 = torch.cat([d1, e1], dim=1)

        out = self.final_conv(d1) # (B,2, F, T)
        return torch.tanh(out)

# ───────────────────────────────────────────────────────────────────────────────
# Updated Frequency Branch & Fusion unchanged
# ───────────────────────────────────────────────────────────────────────────────
class FrequencyDomainBranch(nn.Module):
    def __init__(self, n_fft=1024, hop_length=256):
        super().__init__()
        self.n_fft = n_fft; self.hop_length = hop_length
        window = torch.hann_window(n_fft)
        self.register_buffer('window', window)
        self.unet = ComplexMaskUNet(in_channels=1, base_channels=16)

    def forward(self, x):
        x_wave = x.squeeze(1)
        X = torch.stft(x_wave, n_fft=self.n_fft,
                       hop_length=self.hop_length,
                       window=self.window, return_complex=True)
        real, imag = X.real, X.imag
        mag = torch.abs(X).unsqueeze(1)

        mask = self.unet(mag)
        mr, mi = mask[:,0], mask[:,1]

        real_hat =  real * mr - imag * mi
        imag_hat =  real * mi + imag * mr
        X_hat = torch.complex(real_hat, imag_hat)

        x_hat = torch.istft(X_hat, n_fft=self.n_fft,
                            hop_length=self.hop_length,
                            window=self.window,
                            length=x_wave.size(-1))
        return x_hat.unsqueeze(1)

class GatedFusionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        pad = kernel_size//2
        self.gate = nn.Sequential(
            nn.Conv1d(2,1,kernel_size,padding=pad),
            nn.Sigmoid()
        )
    def forward(self, t, f):
        cat = torch.cat([t,f], dim=1)
        α = self.gate(cat)
        return α*t + (1-α)*f

class HybridPercussionSeparator(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_branch = TimeDomainBranch(channels=32)
        self.freq_branch = FrequencyDomainBranch(n_fft=1024, hop_length=256)
        self.fusion      = GatedFusionModule(kernel_size=7)

    def forward(self, x):
        t = self.time_branch(x)
        f = self.freq_branch(x)
        return self.fusion(t, f)
