# REST Implementation: Next Steps

## Status
✅ **Core Architecture Implemented**
- Temporal VAE (video compression)
- ID-Context Cache (identity + temporal consistency)
- A2V-DiT (28 transformer blocks)
- Flow Matching scheduler (fast convergence)
- Audio conditioning (SpeechAE)
- Asynchronous Streaming Distillation framework

✅ **Project Structure Ready**
- Dataset loaders (with dummy support for testing)
- Config system (YAML-based)
- Logging infrastructure
- Runpod integration

⏳ **Training Scripts** (Placeholders - need full implementation)
- `scripts/train_temporal_vae.py` - VAE training loop
- `scripts/train_teacher.py` - Teacher model (non-streaming)
- `scripts/train_student.py` - Student model with ASD
- `scripts/inference.py` - Real-time generation

---

## What's Working Now

### 1. Models (All importable)
```python
from models import (
    TemporalVAE,           # Video compression
    IDContextCache,        # Streaming attention
    A2VDIT,               # Diffusion transformer
    SpeechAE,             # Audio encoding
    FlowMatcher,          # Fast diffusion
)
```

### 2. Quick Test
```bash
cd rest-training
python -c "from models import TemporalVAE; vae = TemporalVAE(); print('✅ Models loaded')"
```

### 3. Dataset Loading
```python
from datasets import TalkingHeadDataLoader

train_loader, val_loader = TalkingHeadDataLoader.create_loaders(
    data_root="datasets/",
    use_dummy=True,  # Uses synthetic data for testing
    batch_size=8,
)

batch = next(iter(train_loader))
print(batch['video'].shape)  # (8, 3, 16, 64, 64)
```

---

## Implementation Tasks (Priority Order)

### Phase 1: Core Training Loop (High Priority - Days 1-2)

**Task 1.1: VAE Training Loop**
- [ ] Implement `train_temporal_vae.py`
- [ ] Load config, create dataloader
- [ ] VAE forward pass + loss calculation
- [ ] Checkpoint saving
- [ ] Validation metrics (FID, reconstruction quality)
- **Expected output:** `checkpoints/vae_best.pt` (48 hours on A40)

**Task 1.2: Teacher Training Loop**
- [ ] Implement `train_teacher.py`
- [ ] Load VAE encoder (freeze)
- [ ] Create A2V-DiT model
- [ ] Flow Matching loss
- [ ] Non-streaming inference
- **Expected output:** `checkpoints/teacher_best.pt` (96 hours)

**Task 1.3: Student + ASD Training**
- [ ] Implement `train_student.py`
- [ ] ID-Context Cache setup
- [ ] Teacher supervision (KD loss)
- [ ] Asynchronous noise scheduler
- [ ] Streaming inference support
- **Expected output:** `checkpoints/student_best.pt` (120 hours)

### Phase 2: Inference & Benchmarking (High Priority - Days 2-3)

**Task 2.1: Inference Pipeline**
- [ ] Implement `inference.py`
- [ ] Real-time frame generation
- [ ] Audio-video synchronization
- [ ] Lip-sync metrics
- [ ] Output video generation

**Task 2.2: Benchmarking**
- [ ] FVD (Fréchet Video Distance)
- [ ] FID (Face ID consistency)
- [ ] Lip-sync accuracy
- [ ] Latency measurement
- [ ] Compare vs. baselines (Synthesia, HeyGen, etc.)

### Phase 3: Optimization (Medium Priority - Days 3-4)

**Task 3.1: Speed Optimization**
- [ ] Model quantization (FP16 → INT8)
- [ ] Batch inference
- [ ] Streaming generation (chunk-by-chunk)
- [ ] GPU memory profiling

**Task 3.2: Quality Improvements**
- [ ] Hyperparameter tuning
- [ ] Data augmentation
- [ ] Fine-tuning on specific persons

### Phase 4: Real Data Integration (Lower Priority - Days 4+)

**Task 4.1: Dataset Pipeline**
- [ ] Implement VoxCeleb2 downloader
- [ ] Video preprocessing (face detection, alignment)
- [ ] Audio alignment
- [ ] Data validation

---

## Development Roadmap

### Week 1: Get Training Working
```
Day 1-2: VAE training loop functional
Day 2-3: Teacher training loop functional
Day 3-4: Student + ASD training functional
Day 4-5: Inference working end-to-end
Day 5-7: Benchmarking + optimization
```

### Week 2: Scale & Polish
```
Day 8-10: Real dataset integration
Day 10-12: Performance tuning
Day 12-14: Evaluation on benchmarks
```

---

## Runpod Training Timeline

### Estimated ($100 budget on A40 @ $0.29/hr)

**Stage 1: Temporal VAE**
- Time: 48 hours (2 days continuous)
- Cost: $14
- Hardware: 1x A40 GPU
- Checkpoints: `vae_best.pt`

**Stage 2: Teacher (Non-streaming)**
- Time: 96 hours (4 days)
- Cost: $28
- Hardware: 1x A40 GPU
- Checkpoints: `teacher_best.pt`

**Stage 3: Student + ASD (Streaming)**
- Time: 120 hours (5 days)
- Cost: $35
- Hardware: 1x A40 GPU
- Checkpoints: `student_best.pt`

**Remaining Budget: $23** (Testing, inference, fine-tuning)

---

## Current Blockers

None! Ready to implement training loops.

---

## Code Review Checklist

Before pushing each training script:

- [ ] Config loading works
- [ ] Dataloader creates batches correctly
- [ ] Model forward pass runs without error
- [ ] Loss calculation is correct
- [ ] Gradient backward works
- [ ] Checkpoint saving/loading works
- [ ] TensorBoard logging functional
- [ ] Inference pipeline functional
- [ ] Can run for 1 hour without crash
- [ ] GPU memory usage is reasonable (<12GB on A40)

---

## Reference Implementation

Key files to reference:
- `models/temporal_vae.py` - VAE encoder/decoder (complete)
- `models/id_context_cache.py` - ID-Sink + Context-Cache (complete)
- `models/a2v_dit.py` - Diffusion transformer (complete)
- `datasets/talking_head_dataset.py` - Data loading (complete)

These are fully working and ready to use in training loops.

---

## Estimated Effort

| Component | Implementation Time | Difficulty |
|-----------|----------------------|------------|
| VAE training | 2-4 hours | Low |
| Teacher training | 2-4 hours | Low |
| Student + ASD | 4-6 hours | Medium |
| Inference | 2-3 hours | Low |
| Benchmarking | 2-3 hours | Low |
| **Total** | **12-20 hours** | - |

---

## How to Get Started

1. **Pick ONE training script to implement** (recommend: VAE first)
2. Use the config system (YAML)
3. Test with dummy dataset first
4. Push checkpoint to Runpod
5. Monitor with Tensorboard
6. Repeat for next stage

---

## Questions?

- Architecture unclear? See `models/` docstrings
- Config system? Check `configs/training_config.yaml`
- Dataset structure? See `datasets/talking_head_dataset.py`
- Runpod setup? Follow `TRAINING_GUIDE.md`

All core building blocks are ready. Just need to wire them together in training loops! 🚀
