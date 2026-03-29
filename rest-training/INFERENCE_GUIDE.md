# REST Inference & Evaluation Guide

After training the teacher model, use these tools to evaluate quality and generate outputs.

## Quick Start

Run the complete pipeline:

```bash
bash run_inference_pipeline.sh
```

This executes:
1. **Inference** — test teacher on held-out data
2. **Evaluation** — comprehensive quality assessment
3. **Video Generation** — create sample outputs

---

## Individual Scripts

### 1. **Inference** (`scripts/inference.py`)

Test the trained teacher model and compute quality metrics.

```bash
python3 scripts/inference.py \
    --checkpoint-dir checkpoints/ \
    --teacher-ckpt teacher_working.pt \
    --batch-size 4 \
    --num-batches 5
```

**Output:**
- Per-batch metrics:
  - **MSE** — Mean squared error (lower is better)
  - **L2 Distance** — Euclidean distance in latent space
  - **Cosine Similarity** — Angular similarity (higher is better, ideal: 1.0)
  - **Correlation** — Linear correlation between predicted and target

**Interpretation:**
- MSE < 0.001: Excellent
- MSE < 0.01: Good
- MSE < 0.05: Fair
- MSE > 0.05: Poor

---

### 2. **Evaluation** (`scripts/evaluate.py`)

Comprehensive evaluation with detailed report.

```bash
python3 scripts/evaluate.py \
    --checkpoint-dir checkpoints/ \
    --teacher-ckpt teacher_working.pt \
    --batch-size 4 \
    --num-batches 10 \
    --output-report evaluation_report.txt
```

**Metrics Computed:**
- **Latent-space metrics:**
  - MSE, MAE, L2 distance
  - Cosine similarity (with std dev)
  - Correlation

- **Perceptual metrics:**
  - Temporal coherence (frame-to-frame consistency)
  - Spatial consistency (smoothness in latent space)

- **Video metrics** (if available):
  - Video reconstruction MSE

**Output:**
- `evaluation_report.txt` — Detailed report with statistics
- Console output — Real-time summary
- Quality assessment: EXCELLENT/GOOD/FAIR/POOR

---

### 3. **Video Generation** (`scripts/generate_video.py`)

Generate videos from audio using the trained teacher.

```bash
python3 scripts/generate_video.py \
    --checkpoint-dir checkpoints/ \
    --teacher-ckpt teacher_working.pt \
    --output-dir outputs/ \
    --batch-size 2 \
    --num-samples 3
```

**Pipeline:**
```
Audio → [Audio Encoder] → Audio Embeddings
                          ↓
                    [Teacher Network]
                          ↓
                    Predicted Latents
                          ↓
                    [VAE Decoder]
                          ↓
                    Generated Video
```

**Output:**
- `outputs/batch{i}_sample{j}_real.mp4` — Ground truth video
- `outputs/batch{i}_sample{j}_predicted.mp4` — Teacher-generated video

**Comparison:**
Side-by-side comparison shows how well the teacher maps audio to video semantics.

---

## Architecture Overview

### Data Flow

```
Video Input (512×512, 16 frames)
    ↓ [VAE Encoder]
Latents (4, 4, 128, 128) → flattened to (65536, 4)

Audio Input (16000 samples)
    ↓ [Audio Encoder]
Audio Embeddings (100, 256)
    ↓ [Teacher Network]
Predicted Latents (65536, 4)
    ↓ [Comparison with Ground Truth]
Loss = MSE(predicted, ground_truth)
```

### Teacher Network

```
Audio Embedding (B, T_audio, 256)
    ↓ [Dense Layers]
Features (B, T_audio, 256)
    ↓ [Dynamic Expansion]
Expanded (B, 65536, 256)
    ↓ [Projection]
Latents (B, 65536, 4)
```

---

## Evaluation Metrics Explained

### MSE (Mean Squared Error)
- **Range:** [0, ∞)
- **Lower is better**
- **Interpretation:** Average pixel-space error per latent dimension

### Cosine Similarity
- **Range:** [-1, 1]
- **Higher is better** (ideal: 1.0)
- **Interpretation:** Angular similarity in latent space
- **1.0** = identical direction
- **0.0** = orthogonal
- **-1.0** = opposite direction

### Correlation
- **Range:** [-1, 1]
- **Higher is better** (ideal: 1.0)
- **Interpretation:** Linear relationship between predicted and target
- Shows if teacher captures trend correctly

### Temporal Coherence
- **Lower is better**
- Measures frame-to-frame consistency
- High values = jerky/incoherent videos

### Spatial Consistency
- **Higher is better**
- Measures smoothness in latent space
- Low values = inconsistent across batch

---

## Troubleshooting

### "Teacher checkpoint not found"
```
❌ Teacher checkpoint not found at checkpoints/teacher_working.pt
```

**Solution:**
```bash
# Check if checkpoint exists
ls -la checkpoints/

# If not, train first:
bash train.sh --working
```

### "VAE checkpoint not found"
```
⚠️  VAE checkpoint not found, using random init
```

**Solution:**
- This is OK for testing (uses random VAE)
- For real evaluation, provide `checkpoints/vae_best.pt`
- Results will be nonsensical but script will run

### Low quality metrics
```
Overall Quality: POOR
Mean MSE: 0.150
```

**Possible causes:**
1. Teacher didn't converge (increase training epochs)
2. Not enough training data
3. Model architecture mismatch

**Solutions:**
```bash
# Retrain with more epochs
python3 scripts/train_teacher_working.py --epochs 100

# Or check if latent dimensions match
python3 scripts/diagnose_shapes.py
```

---

## Deployment Workflow

If evaluation passes quality thresholds:

1. **Save trained checkpoint** (already done):
   ```bash
   cp checkpoints/teacher_working.pt checkpoints/teacher_production.pt
   ```

2. **Load in production:**
   ```python
   from train_teacher_working import DynamicTeacher
   
   teacher = DynamicTeacher(audio_dim=256, latent_dim=4)
   teacher.load_state_dict(torch.load("teacher_production.pt"))
   teacher.eval()
   ```

3. **Inference:**
   ```python
   with torch.no_grad():
       audio_emb = audio_encoder(audio)  # (B, T_audio, 256)
       pred_latents = teacher(audio_emb, target_seq_len=65536)
       video = vae.decode(pred_latents)  # (B, 3, T, H, W)
   ```

---

## Next Steps

### If quality is GOOD/EXCELLENT:
- ✅ Deploy to production
- ✅ Use for real-time inference
- ✅ Consider quantization/pruning for speed

### If quality is FAIR:
- 🔄 Try increasing training epochs
- 🔄 Use more diverse training data
- 🔄 Try different architecture (add more layers)

### If quality is POOR:
- ⚠️ Check data alignment (audio-video sync)
- ⚠️ Verify latent dimensions match
- ⚠️ Debug with `scripts/train_teacher_debug.py`

---

## References

- **MSE interpretation:** Common ML metric, lower is better
- **Cosine similarity:** Measures alignment in high-dimensional space
- **Temporal coherence:** Ensures smooth transitions frame-to-frame
- **VAE latent space:** Learned compressed representation of video

For more details, see `TEACHER_TRAINING_FIX.md` and training logs.
