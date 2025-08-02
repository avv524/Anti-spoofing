import torch


def collate_fn(batch):
    audios = [item["data_object"] for item in batch]
    
    # Simple padding to max length in batch
    max_len = max(audio.size(0) for audio in audios)
    
    padded_audios = []
    for audio in audios:
        if audio.size(0) < max_len:
            # Pad with zeros
            pad_len = max_len - audio.size(0)
            audio = torch.nn.functional.pad(audio, (0, pad_len))
        padded_audios.append(audio)
    
    return {
        "data_object": torch.stack(padded_audios),
        "labels": torch.stack([item["labels"] for item in batch]),
        "file_id": [item["file_id"] for item in batch]  # For submission
    } 