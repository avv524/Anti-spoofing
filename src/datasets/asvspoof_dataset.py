import torch
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset
import random


class ASVSpoofDataset(Dataset):
    
    def __init__(self, data_root, subset="train", sample_rate=16000, max_audio_length=6.0, shuffle_index=None, **kwargs):
        self.data_root = Path(data_root)
        self.subset = subset
        self.sample_rate = sample_rate
        self.max_audio_length = max_audio_length
        self.max_audio_samples = int(max_audio_length * sample_rate) if max_audio_length else None
        self.samples = self._load_protocol()
        
        if shuffle_index:
            random.seed(42)
            random.shuffle(self.samples)
    
    def _load_protocol(self):
        """Load protocol file and create simple sample list"""
        protocol_files = {
            "train": "ASVspoof2019.LA.cm.train.trn.txt",
            "dev": "ASVspoof2019.LA.cm.dev.trl.txt", 
            "eval": "ASVspoof2019.LA.cm.eval.trl.txt"
        }
        
        protocol_path = self.data_root / "ASVspoof2019_LA_cm_protocols" / protocol_files[self.subset]
        
        samples = []
        with open(protocol_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    file_id = parts[1]
                    label = 1 if parts[4] == "bonafide" else 0 
                    audio_path = self.data_root / f"ASVspoof2019_LA_{self.subset}" / "flac" / f"{file_id}.flac"
                    
                    if audio_path.exists():
                        samples.append({
                            "file_id": file_id,
                            "audio_path": str(audio_path),
                            "label": label
                        })
        
        return samples
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio, sr = torchaudio.load(sample["audio_path"])
        
        # Simple resampling if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)
        
        # Convert to mono if needed
        if audio.size(0) > 1:
            audio = audio.mean(dim=0)
        else:
            audio = audio.squeeze(0)
        
        # Truncate audio if it's too long
        if self.max_audio_samples and audio.size(0) > self.max_audio_samples:
            audio = audio[:self.max_audio_samples]
        
        return {
            "data_object": audio,
            "labels": torch.tensor(sample["label"], dtype=torch.long),
            "file_id": sample["file_id"]
        }
    
    def __len__(self):
        return len(self.samples) 