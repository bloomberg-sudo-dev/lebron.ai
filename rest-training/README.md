# REST: Real-time End-to-end Streaming Talking Head Generation

Diffusion-based real-time streaming talking head generation with ID-Context Caching and Asynchronous Streaming Distillation.

**Paper:** REST (Wang et al., USTC + iFLYTEK, Dec 2025)

## Quick Start

### Local Development
```bash
git clone https://github.com/bloomberg-sudo-dev/lebron.ai.git
cd lebron.ai/rest-training
bash scripts/setup_local.sh
```

### Runpod Training
```bash
# 1. Create Runpod Pod (A40 GPU recommended)
# 2. SSH into pod
# 3. Run setup
bash scripts/runpod_setup.sh

# 4. Training starts automatically
# Check logs: tail -f logs/training.log
```

## Architecture Overview

### Core Components

1. **Temporal VAE** (`models/temporal_vae.py`)
   - Compresses video: 32×32×8 pixels per token
   - 8-32x computation reduction
   - Enables real-time latent diffusion

2. **ID-Context Cache** (`models/id_context_cache.py`)
   - ID-Sink: Reference face embedding (constant)
   - Context-Cache: Previous chunk context (temporal flow)
   - Chunk-by-chunk autoregressive generation
   - Maintains identity + temporal consistency

3. **A2V-DiT** (`models/a2v_dit.py`)
   - 28 transformer blocks
   - Self-Attention with ID-Context Cache
   - 3D Full-Attention for conditioning
   - Frame-level 2D Cross-Attention for audio

4. **Asynchronous Streaming Distillation** (`trainers/asd_trainer.py`)
   - Non-streaming teacher model
   - Asynchronous chunk-wise noise scheduler
   - Streaming student supervised by teacher
   - Mitigates error accumulation

## Training Pipeline

### Stage 1: Temporal VAE Training
```bash
python scripts/train_temporal_vae.py \
  --config configs/vae_config.yaml \
  --gpus 1 \
  --batch-size 16
```

### Stage 2: Teacher Model Training (Non-streaming)
```bash
python scripts/train_teacher.py \
  --config configs/training_config.yaml \
  --gpus 1 \
  --batch-size 8
```

### Stage 3: Student Model Training (Streaming with ID-Context)
```bash
python scripts/train_student.py \
  --config configs/training_config.yaml \
  --gpus 1 \
  --batch-size 8
```

### Inference
```bash
python scripts/inference.py \
  --checkpoint checkpoints/student_best.pt \
  --reference-image path/to/reference.jpg \
  --audio-file path/to/speech.wav \
  --output-video output.mp4
```

## Datasets

Automatically downloaded via `scripts/download_datasets.py`:

- **VoxCeleb2** (talking head videos)
- **RAVDESS** (speech + video)
- **Custom Talking Head Dataset** (prepared for training)

Total: ~100GB for full training

## Project Structure

```
rest-training/
├─ models/
│  ├─ __init__.py
│  ├─ temporal_vae.py        (VAE encoder/decoder)
│  ├─ id_context_cache.py    (ID-Sink + Context-Cache)
│  ├─ a2v_dit.py             (DiT backbone)
│  ├─ flow_matching.py       (FM scheduler)
│  └─ audio_encoder.py       (SpeechAE + Whisper)
│
├─ trainers/
│  ├─ __init__.py
│  ├─ base_trainer.py
│  ├─ vae_trainer.py
│  ├─ asd_trainer.py         (Asynchronous Streaming Distillation)
│  └─ utils.py
│
├─ scripts/
│  ├─ setup_local.sh          (local setup)
│  ├─ runpod_setup.sh         (Runpod auto-setup)
│  ├─ download_datasets.py    (dataset downloader)
│  ├─ train_temporal_vae.py
│  ├─ train_teacher.py
│  ├─ train_student.py
│  ├─ inference.py
│  └─ benchmark.py
│
├─ configs/
│  ├─ vae_config.yaml
│  ├─ training_config.yaml
│  └─ inference_config.yaml
│
├─ notebooks/
│  └─ training.ipynb          (Jupyter for Runpod)
│
├─ datasets/
│  ├─ __init__.py
│  ├─ talking_head_dataset.py
│  └─ audio_processor.py
│
├─ utils/
│  ├─ logger.py
│  ├─ checkpoint.py
│  └─ metrics.py
│
└─ tests/
   ├─ test_models.py
   └─ test_training.py
```

## Training Timeline

**A40 GPU ($0.29/hr, $100 budget = 344 hours):**

| Stage | Hours | Cost | Description |
|-------|-------|------|-------------|
| VAE Training | 48 | $14 | Temporal compression learning |
| Teacher Training | 96 | $28 | Non-streaming baseline |
| Student Training | 120 | $35 | Streaming with ASD supervision |
| Fine-tuning | 48 | $14 | Quality optimization |
| Testing/Inference | 32 | $9 | Validation + benchmarks |
| **Total** | **344** | **$100** | Full training cycle |

## Hyperparameters

See `configs/training_config.yaml` for details:

- **VAE**: 8x8x4 compression, KL weight: 0.00001
- **Teacher**: LR 1e-4, batch 8, 100k steps
- **Student**: LR 5e-5, batch 8, 150k steps
- **ASD**: Asynchronous scheduler, distillation weight: 0.1

## Results

Target metrics (based on REST paper):

- **Inference latency:** <500ms end-to-end
- **FVD:** <60 (video quality)
- **Lip-sync accuracy:** >95%
- **Identity consistency:** >0.9 (cosine similarity)

## Monitoring

Real-time monitoring via:

```bash
# Terminal 1: Training logs
tail -f logs/training.log

# Terminal 2: TensorBoard
tensorboard --logdir logs/tensorboard

# Terminal 3: GPU monitoring
watch -n 1 nvidia-smi
```

## Runpod Commands

```bash
# Start training (SSH into pod)
bash /workspace/rest-training/scripts/runpod_setup.sh

# Resume training
python scripts/train_student.py --resume --checkpoint checkpoints/student_latest.pt

# Download results
scp -r root@<pod-ip>:/workspace/rest-training/checkpoints ./checkpoints
```

## References

- Paper: REST (arXiv:2512.11229v1)
- Flow Matching: Lipman et al., 2022
- Temporal VAE: LTX-Video (HaCohen et al., 2024)
- DiT: Peebles & Xie, 2022

## License

MIT

## Contact

bloomberg-sudo-dev (GitHub)
