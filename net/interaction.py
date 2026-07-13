from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

PART_NAMES = ('head', 'torso', 'left_arm', 'right_arm', 'left_leg', 'right_leg')
NTU_BODY_PARTS = {
    'head': [2, 3], 'torso': [0, 1, 20],
    'left_arm': [4, 5, 6, 7, 21, 22], 'right_arm': [8, 9, 10, 11, 23, 24],
    'left_leg': [12, 13, 14, 15], 'right_leg': [16, 17, 18, 19],
}

class DynamicBodyInteraction(nn.Module):
    """Computes a per-sample dynamic adjacency matrix from raw skeleton input."""
    def __init__(self):
        super().__init__()
        self.num_parts = len(PART_NAMES)
        self.joint_indices = [NTU_BODY_PARTS[name] for name in PART_NAMES]
        
    def forward(self, raw_x):
        # raw_x: (N, C, T, V, M)
        N, C, T, V, M = raw_x.shape
        x = raw_x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        
        # Aggregate joints to body-part centroids
        parts = []
        for indices in self.joint_indices:
            part_centroid = x[:, :, :, indices].mean(dim=-1, keepdim=True)
            parts.append(part_centroid)
        
        centroids = torch.cat(parts, dim=-1)  # (N*M, C, T, P=6)
        
        # Compute velocity via diff along T
        velocity = centroids[:, :, 1:, :] - centroids[:, :, :-1, :]  # (N*M, C, T-1, P)
        
        # Compute speed = L2 norm over C
        speed = velocity.norm(dim=1)  # (N*M, T-1, P)
        
        # Compute pairwise Pearson correlation
        speed_centered = speed - speed.mean(dim=1, keepdim=True)
        speed_normed = speed_centered / speed_centered.norm(dim=1, keepdim=True).clamp(min=1e-8)
        
        # Correlation: (N*M, P, T-1) @ (N*M, T-1, P) -> (N*M, P, P)
        corr = torch.bmm(speed_normed.transpose(1, 2), speed_normed)
        
        # Clamp negatives to 0
        corr = corr.clamp(min=0)
        
        # Add self-loops
        eye = torch.eye(self.num_parts, device=corr.device).unsqueeze(0)
        corr = corr + eye
        
        # Row-normalize
        corr = corr / corr.sum(dim=-1, keepdim=True).clamp(min=1.0)
        
        return corr

class TemporalEnergyAttention(nn.Module):
    """Computes soft temporal attention weights from raw skeleton motion energy."""
    def __init__(self, sigma=3.0, window=15):
        super().__init__()
        x = torch.arange(window, dtype=torch.float32) - window // 2
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()
        self.register_buffer('kernel', kernel.view(1, 1, -1))
        self.window = window
        
    def forward(self, raw_x):
        # raw_x: (N, C, T, V, M)
        N, C, T, V, M = raw_x.shape
        x = raw_x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        
        # Compute velocity: diff along T
        velocity = x[:, :, 1:, :] - x[:, :, :-1, :]  # (N*M, C, T-1, V)
        
        # Compute per-frame energy
        energy = velocity.norm(dim=1).sum(dim=-1)  # (N*M, T-1)
        
        # Smooth with the Gaussian kernel using F.conv1d
        e_unsqueezed = energy.unsqueeze(1)
        pad = self.window // 2
        e_padded = F.pad(e_unsqueezed, (pad, pad), mode='reflect')
        smoothed = F.conv1d(e_padded, self.kernel).squeeze(1)  # (N*M, T-1)
        
        # Normalize to [0, 1] per sample, guarding against zero-motion
        # (e.g. padding person M=2 in NTU is all zeros -> energy is 0)
        e_min = smoothed.min(dim=1, keepdim=True)[0]
        e_max = smoothed.max(dim=1, keepdim=True)[0]
        denom = e_max - e_min
        # If a sample has zero motion (all-zeros skeleton), denom is 0.
        # Replace with 1.0 so normalized becomes 0.0 (uniform attention).
        safe_denom = torch.where(denom > 1e-6, denom, torch.ones_like(denom))
        normalized = (smoothed - e_min) / safe_denom
        
        # Add baseline of 0.5 so even low-motion frames contribute
        attn = 0.5 + 0.5 * normalized
        
        return attn

if __name__ == '__main__':
    raw_x = torch.randn(2, 3, 300, 25, 2)
    dyn_int = DynamicBodyInteraction()
    corr = dyn_int(raw_x)
    print("DynamicBodyInteraction output shape:", corr.shape)
    assert corr.shape == (4, 6, 6)
    
    temp_attn = TemporalEnergyAttention()
    attn = temp_attn(raw_x)
    print("TemporalEnergyAttention output shape:", attn.shape)
    assert attn.shape == (4, 299)
    assert torch.isfinite(attn).all(), "NaN/Inf in normal attention!"
    print("[PASS] normal input")

    # Test zero-motion (simulates NTU padding person M=2 = all zeros)
    raw_zero = torch.zeros(2, 3, 300, 25, 2)
    attn_zero = temp_attn(raw_zero)
    assert torch.isfinite(attn_zero).all(), "NaN/Inf in zero-motion attention!"
    print("[PASS] zero-motion input (NaN bug fixed)")

    # Test mixed: person 1 has motion, person 2 is all zeros
    raw_mixed = torch.randn(2, 3, 300, 25, 2)
    raw_mixed[:, :, :, :, 1] = 0.0  # person 2 is padding
    attn_mixed = temp_attn(raw_mixed)
    assert torch.isfinite(attn_mixed).all(), "NaN/Inf in mixed attention!"
    print("[PASS] mixed input (real + padding person)")

