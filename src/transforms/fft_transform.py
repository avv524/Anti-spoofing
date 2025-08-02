import torch
import torch.nn as nn
import torchaudio.transforms as T


class FFTSpectrogram(nn.Module):
    """
    STFT feature extraction.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        win_length: int = 400,            # 25ms * 16000 = 400 samples
        hop_length: int = 160,            # 10ms * 16000 = 160 samples
        n_fft: int = 1024,
        power: float = 2.0,
        normalize: bool = False
    ):
        super().__init__()
        self.win_length = win_length
        self.hop_length = hop_length
        self.power = power

        self.spectrogram = T.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window_fn=torch.hamming_window,
            power=power,
            normalized=normalize,
            center=True,
            pad_mode="reflect"
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = self.spectrogram(waveform)
        if spec.dim() == 3:
            spec = spec.unsqueeze(1)
        return spec


FFTTransform = FFTSpectrogram
FFTBatchTransform = FFTSpectrogram