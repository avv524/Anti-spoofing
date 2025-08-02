# Voice Anti-spoofing with LCNN

This project implements a Light CNN (LCNN) for voice anti-spoofing on the ASVSpoof 2019 Logical Access (LA) dataset.

### LCNN Model:
- **Input**: Raw FFT spectrum
- **Dropout**: 0.75
- **Output**: Binary classification (bonafide vs spoof)

### FFT Front-end Configuration:
- **Window Length**: 400
- **Hop Length**: 160
- **N_fft**: 1024
- **Max Audio Length**: 6.0s
- **Power**: 2.0

### Training Recipe:
- **Optimizer**: Adam (β₁=0.9, β₂=0.999, ε=10⁻⁸)
- **Learning Rate**: 3×10⁻⁴, reduced by 0.5 every 20 epochs
- **Batch Size**: 8 or 64

## **Quick Start**

# Install dependencies
pip install -r requirements.txt

# Change paths to yours

# LCNN + Cross-Entropy Loss
python train.py -cn=asvspoof_lcnn

### Test Best Model
python inference.py -cn=asvspoof_inference

### Generate Submission File
python scripts/generate_submission.py \
  --model_path saved/asvspoof_lcnn_baseline/model_best.pth \
  --data_path "your path" \
  --output "something".csv

### Target Results:
- **Cross-Entropy**: ~5.3% EER
- i trained this model on RTX3080 for about 5 hrs to obtain 5.3%

## **Acknowledgments**

Based on the PyTorch Project Template by [Blinorot](https://github.com/Blinorot/pytorch_project_template).
