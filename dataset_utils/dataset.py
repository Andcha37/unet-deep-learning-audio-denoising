import random
from pathlib import Path
from torch.utils.data import Dataset
from preprocess_utils.audio_preprocess import sound_load, audio_merge, audio_merge_eval, extract_stft_features

class FullDataset(Dataset):
    def __init__(self, clean_seg_dir, noise_dir, target_sr=16000, target_len=65280, transform=None, is_train=True):
        self.clean_files = sorted(list(Path(clean_seg_dir).rglob("*.wav")))
        self.noise_files = sorted(list(Path(noise_dir).rglob("*.wav")))
        self.target_sr = target_sr
        self.target_len = target_len
        self.transform = transform
        self.is_train = is_train

        if len(self.clean_files) == 0 or len(self.noise_files) == 0:
            raise ValueError("오디오 파일을 찾을 수 없습니다. 경로를 재확인해주세요.")

    def __len__(self):
        return len(self.clean_files)
    
    def __getitem__(self, idx):
        clean_path = self.clean_files[idx]
        clean_seg, _ = sound_load(clean_path, self.target_sr)

        if self.is_train:
            noise_path = random.choice(self.noise_files)
            noise_wav, _ = sound_load(noise_path, self.target_sr)
            c_seg_processed, m_seg_processed = audio_merge(clean_seg, noise_wav)
        else:
            fixed_noise_idx = idx % len(self.noise_files)
            noise_path = self.noise_files[fixed_noise_idx]
            noise_wav, _ = sound_load(noise_path, self.target_sr)
            c_seg_processed, m_seg_processed = audio_merge_eval(clean_seg, noise_wav)

        c_mag, c_phase = extract_stft_features(c_seg_processed)
        m_mag, m_phase = extract_stft_features(m_seg_processed)

        data = {
            "label": c_mag,
            "input": m_mag,
            "clean_phase": c_phase,
            "noisy_phase": m_phase
        }

        return data