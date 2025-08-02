import argparse
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.datasets.asvspoof_dataset import ASVSpoofDataset
from src.datasets.collate import collate_fn
from src.model.lcnn_model import LCNNModel
from torch.utils.data import DataLoader


def load_model(model_path, device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint = torch.load(model_path, map_location=device)
    
    if "state_dict" in checkpoint:
        model_state = checkpoint["state_dict"]
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    elif "model" in checkpoint:
        model_state = checkpoint["model"]
    else:
        model_state = checkpoint
    
    model = LCNNModel(
        n_class=2,
        input_channels=1,
        dropout_rate=0.75
    )
    
    try:
        model.load_state_dict(model_state)
        print("Model state dict loaded successfully!")
    except RuntimeError as e:
        print(f"Error loading state dict: {e}")
        raise e
    
    model.to(device)
    model.eval()
    
    return model, device


def generate_predictions(model, dataset, batch_transform, device, batch_size=8):
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True if device.startswith('cuda') else False,
        collate_fn=collate_fn
    )
    
    predictions = {}
    
    print(f"Processing {len(dataset)} samples...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating predictions"):
            raw_audio = batch["data_object"]
            raw_audio = raw_audio.to(device)
            features = batch_transform(raw_audio)
            file_ids = batch["file_id"]
            outputs = model(features)
            logits = outputs["logits"]
            log_probs = torch.log_softmax(logits, dim=1)
            scores = log_probs[:, 1] - log_probs[:, 0]
            for file_id, score in zip(file_ids, scores.cpu().numpy()):
                predictions[file_id] = float(score)
    
    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Generate ASVSpoof2019 submission file with torchaudio pipeline"
    )
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to ASVSpoof2019 dataset")
    parser.add_argument("--output", type=str, default="submission.csv",
                       help="Output CSV file path (hse_email.csv)")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for inference")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--subset", type=str, default="eval",
                       help="Dataset to use")
    parser.add_argument("--frontend", type=str, default="fft", choices=["fft"],
                       help="Frontend type")
    
    args = parser.parse_args()
    model, device = load_model(args.model_path, args.device)
    
    dataset = ASVSpoofDataset(
        data_root=args.data_path,
        subset=args.subset,
        sample_rate=16000,
        shuffle_index=False
    )
    print(f"Dataset loaded with {len(dataset)} samples")
    from src.transforms.fft_transform import FFTSpectrogram
    batch_transform = FFTSpectrogram(
        sample_rate=16000,
        win_length=400,
        hop_length=160,
        n_fft=1024,
        power=2.0,
        normalize=False,
    ).to(device)
    
    print("Generating predictions...")
    predictions = generate_predictions(
        model=model,
        dataset=dataset,
        batch_transform=batch_transform,
        device=device,
        batch_size=args.batch_size
    )
    
    submission_data = []
    for file_id, prediction in predictions.items():
        submission_data.append({
            "file_id": file_id,
            "prediction": prediction
        })
    
    df = pd.DataFrame(submission_data)
    df = df.sort_values("file_id")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main() 