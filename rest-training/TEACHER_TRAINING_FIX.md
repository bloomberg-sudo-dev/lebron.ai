# Teacher Training Fix - Complete Solution

## The Problem

**Shape mismatch error:**
```
AssertionError: Shape mismatch: pred=torch.Size([2, 16, 4]) vs target=torch.Size([2, 65536, 4])
```

**Root cause:** 
- VAE outputs latents with shape `(B, 4, T, H, W)` → when flattened: `(B, T×H×W, 4)` = `(B, 65536, 4)`
- Audio encoder outputs `(B, T_audio, 256)` where `T_audio ≈ 16` tokens
- Teacher was NOT expanding the 16 tokens to 65536 tokens

## The Solution

### `train_teacher_fixed.py`

Three-stage architecture:

**Stage 1: Audio Processing**
```python
self.audio_encoder = nn.Sequential(
    nn.Linear(audio_dim, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
)
```
Maps audio features through dense layers.

**Stage 2: Sequence Expansion** ⭐ KEY FIX
```python
# If input is (B, 16, 256), expand to (B, 65536, 256)
if T_audio != self.target_seq_len:
    repeat_factor = max(1, self.target_seq_len // T_audio)
    x = x.repeat(1, repeat_factor, 1)  # Repeat tokens
    
    # Fine-tune with interpolation if needed
    if x.shape[1] != self.target_seq_len:
        x = interpolate(x, size=target_seq_len, mode='linear')
```

**Stage 3: Latent Projection**
```python
latents = self.latent_proj(x)  # (B, target_seq_len, 4)
```

## How to Use

### Quick Start

```bash
cd rest-training
bash train.sh
```

Or directly:

```bash
python3 scripts/train_teacher_fixed.py \
    --epochs 50 \
    --batch-size 2 \
    --lr 1e-3 \
    --data-root datasets/ \
    --checkpoint-dir checkpoints/
```

### With Custom Parameters

```bash
python3 scripts/train_teacher_fixed.py \
    --epochs 100 \
    --batch-size 4 \
    --lr 2e-3
```

## Verification

The script includes shape assertions:

```python
if pred_latents.shape != latents_target.shape:
    print(f"❌ Shape mismatch: {pred_latents.shape} vs {latents_target.shape}")
    sys.exit(1)
```

If shapes don't match, it **fails immediately** with clear output (no silent broadcasting).

## Output

```
✅ Checkpoint saved: checkpoints/teacher_fixed.pt
```

Load for inference:

```python
teacher = ExpandingTeacher(audio_dim=256, latent_dim=4, target_seq_len=65536)
teacher.load_state_dict(torch.load("checkpoints/teacher_fixed.pt"))
teacher.eval()

# Predict
with torch.no_grad():
    audio_emb = audio_encoder(audio)  # (B, T_audio, 256)
    latents_pred = teacher(audio_emb)  # (B, 65536, 4)
```

## Why This Works

1. **Explicit expansion:** Token repeat + interpolation guarantees output shape
2. **Multi-stage:** Separates audio processing, expansion, and projection concerns
3. **No assumptions:** Dynamically detects VAE output dimensions
4. **Fail-fast:** Shape assertions prevent silent errors

## Architecture Diagram

```
Audio Input (B, T_audio, 384)
    ↓ [Audio Encoder - frozen]
Audio Emb (B, T_audio, 256)
    ↓ [Stage 1: Process]
Features (B, T_audio, 256)
    ↓ [Stage 2: Expand] ⭐ KEY
Expanded (B, target_seq_len, 256)
    ↓ [Stage 3: Project to latent]
Latents (B, target_seq_len, 4)
    ↓ [Compare with VAE latents]
Loss = MSE(pred, target)
```

## Files Changed

- `scripts/train_teacher_fixed.py` — Complete rewrite with explicit expansion
- `train.sh` — Updated to use fixed version
- `scripts/train_teacher_robust.py` — Previous attempt (kept for reference)

## Next Steps

1. Run `bash train.sh`
2. Monitor loss (should decrease)
3. Once trained, use for speech2latent inference

Done! 🚀
