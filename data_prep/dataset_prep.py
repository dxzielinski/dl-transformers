"""
It takes a splitted dataset of audio files and creates a new datasets
with spectrograms.
"""

import os
import torch
import torchaudio
import noisereduce as nr
from torchaudio.transforms import MelSpectrogram


class AudioProcessing(torch.nn.Module):
    def __init__(self, input_freq=16000, n_fft=1024, n_mel=80, device="cuda"):
        """
        Args:
            input_freq (int): The sample rate of the input waveform.
            n_fft (int): The number of FFT bins used for the spectrogram.
            n_mel (int): The number of mel filter banks.
        """
        super().__init__()
        self.input_freq = input_freq
        self.mel_spec = MelSpectrogram(
            sample_rate=input_freq, n_fft=n_fft, n_mels=n_mel
        )

    def forward(self, waveform: torch.Tensor, sr) -> torch.Tensor:
        device = waveform.device
        audio = waveform.cpu().numpy()
        sr = sr.cpu().numpy()
        noise_profile = audio[: int(sr * 0.5)]
        reduced_noise = nr.reduce_noise(y=audio, sr=sr, y_noise=noise_profile)
        reduced_noise = (
            torch.from_numpy(reduced_noise)
            .float()
            .to(device=device, dtype=torch.float32)
        )
        sr = torch.from_numpy(sr).int().to(device=device, dtype=torch.int16)
        if reduced_noise.numel() == 0 or reduced_noise.size(-1) == 0:
            raise ValueError("The input waveform is empty after noise reduction.")
        trigger_level = 0.1
        trigger_time = 0.1
        front_trimmed = torchaudio.functional.vad(
            reduced_noise, sr, trigger_level=trigger_level, trigger_time=trigger_time
        )
        if front_trimmed.numel() == 0 or front_trimmed.size(-1) == 0:
            front_trimmed = reduced_noise
        flipped_waveform = torch.flip(front_trimmed, [1])
        trimmed_waveform = torchaudio.functional.vad(
            flipped_waveform, sr, trigger_level=trigger_level, trigger_time=trigger_time
        )
        trimmed_waveform = torch.flip(trimmed_waveform, [1])
        if trimmed_waveform.numel() == 0 or trimmed_waveform.size(-1) == 0:
            trimmed_waveform = front_trimmed
        mel = self.mel_spec(trimmed_waveform)
        return mel


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = AudioProcessing().to(device=device, dtype=torch.float32)
    pipeline.eval()
    input_base_dir = "./data-audio"
    output_base_dir = "./data-mel-spectrograms"
    os.makedirs(output_base_dir, exist_ok=True)

    splits = ["train", "val", "test"]

    for split in splits:
        input_split_dir = os.path.join(input_base_dir, split)
        output_split_dir = os.path.join(output_base_dir, split)
        os.makedirs(output_split_dir, exist_ok=True)
        print(f"Processing {split} split...")
        for label in os.listdir(input_split_dir):
            print(f"Processing label: {label}")
            input_label_dir = os.path.join(input_split_dir, label)
            if not os.path.isdir(input_label_dir):
                continue

            output_label_dir = os.path.join(output_split_dir, label)
            os.makedirs(output_label_dir, exist_ok=True)
            for filename in os.listdir(input_label_dir):
                if not filename.lower().endswith(".wav"):
                    continue

                wav_path = os.path.join(input_label_dir, filename)
                waveform, sample_rate = torchaudio.load(wav_path, normalize=True)
                waveform = waveform.to(device=device, dtype=torch.float32)
                sample_rate = torch.tensor(sample_rate, dtype=torch.int16).to(
                    device=device
                )
                print(f"Processing: {filename}")
                with torch.no_grad():
                    mel_spec = pipeline(waveform, sample_rate)

                output_filename = filename.replace(".wav", ".pt")
                output_path = os.path.join(output_label_dir, output_filename)
                torch.save(mel_spec.cpu(), output_path)
