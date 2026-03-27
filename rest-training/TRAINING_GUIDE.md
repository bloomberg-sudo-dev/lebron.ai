# REST Training Guide

## Quick Start on Runpod

### 1. Create Runpod Pod
- Go to [runpod.io](https://runpod.io)
- Select **GPU Pod** → **A40** ($0.29/hr recommended for budget)
- Image: **pytorch/pytorch:2.0-cuda11.8-devel-ubuntu22.04**
- Disk: **100GB**
- Click **CONNECT**

### 2. SSH into Pod
```bash
# Copy SSH command from Runpod console, e.g.:
ssh -p 12345 root@connect.runpod.io
```

### 3. Run Setup Script
```bash
cd /workspace
git clone https://github.com/bloomberg-sudo-dev/lebron.ai.git
cd lebron.ai/rest-training
bash scripts/runpod_setup.sh
```

**That's it!** Training will start automatically.

---

## Training Stages

### Stage 1: Temporal VAE (48 hours)
Learns to compress video to compact latent space (32x32x8 compression)

```bash
python scripts/train_temporal_vae.py \
  --config configs/vae_config.yaml \
  --epochs 50 \
  --batch-size 16
```

**Checkpoint:** `checkpoints/vae_best.pt`

### Stage 2: Teacher Model (96 hours)
Non-streaming baseline for Asynchronous Streaming Distillation

```bash
python scripts/train_teacher.py \
  --config configs/training_config.yaml \
  --vae-checkpoint checkpoints/vae_best.pt \
  --epochs 100 \
  --batch-size 8
```

**Checkpoint:** `checkpoints/teacher_best.pt`

### Stage 3: Student Model (120 hours)
Streaming model with ID-Context Cache supervised by teacher

```bash
python scripts/train_student.py \
  --config configs/training_config.yaml \
  --vae-checkpoint checkpoints/vae_best.pt \
  --teacher-checkpoint checkpoints/teacher_best.pt \
  --epochs 150 \
  --batch-size 8
```

**Checkpoint:** `checkpoints/student_best.pt`

---

## Monitoring Training

### Real-time Logs
```bash
# In Runpod terminal
tail -f logs/student_training.log

# Or use Runpod Jupyter
# Navigate to http://your-pod-ip:8888
```

### Tensorboard
```bash
tensorboard --logdir logs/tensorboard --port 6006
# Access at http://your-pod-ip:6006
```

### GPU Monitoring
```bash
watch -n 1 nvidia-smi
```

---

## Budget Breakdown ($100 on A40)

| Stage | Hours | Cost | Time |
|-------|-------|------|------|
| VAE | 48 | $14 | 2 days |
| Teacher | 96 | $28 | 4 days |
| Student | 120 | $35 | 5 days |
| Inference/Testing | 80 | $23 | 3 days |
| **Total** | **344** | **$100** | **~14 days continuous** |

---

## Running Inference

### After Training Completes
```bash
python scripts/inference.py \
  --checkpoint checkpoints/student_best.pt \
  --reference-image path/to/reference.jpg \
  --audio-file path/to/speech.wav \
  --output-video output.mp4 \
  --num-frames 64
```

### Test on Sample
```bash
python scripts/inference.py \
  --checkpoint checkpoints/student_best.pt \
  --use-dummy  # Uses dummy reference + audio
```

---

## Troubleshooting

### Out of Memory
If CUDA OOM error occurs:
1. Reduce batch size: `--batch-size 4` (instead of 8)
2. Reduce frame size: `frame_size: [32, 32]` in config
3. Use gradient accumulation: `--accumulation-steps 2`

### Slow Training
- Check GPU usage: `nvidia-smi` should show high utilization
- If low utilization, increase batch size
- Use mixed precision: `--mixed-precision fp16`

### Connection Lost
Runpod pods disconnect after inactivity. To keep training:
```bash
# Use nohup or screen
nohup bash scripts/runpod_setup.sh > training.log 2>&1 &

# Or use tmux
tmux new-session -d -s training 'bash scripts/runpod_setup.sh'
```

---

## After Training

### Download Checkpoints
```bash
# From local machine
scp -r root@<pod-ip>:/workspace/rest-training/checkpoints ./checkpoints
scp -r root@<pod-ip>:/workspace/rest-training/logs ./logs
```

### Save to GitHub
```bash
cd lebron.ai
git add -A
git commit -m "Add trained checkpoints"
git push origin main
```

### Shutdown Pod
When done, **stop the pod** on runpod.io to avoid charges.

---

## Next Steps

1. Evaluate on test set: `python scripts/benchmark.py`
2. Compare with baselines (Synthesia, HeyGen, D-ID)
3. Fine-tune on specific person/domain
4. Deploy inference API

---

## Support

- Errors/issues: Check `logs/*.log`
- Code: `models/` and `scripts/`
- Config: `configs/training_config.yaml`

See README.md for full documentation.
