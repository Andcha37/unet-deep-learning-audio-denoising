import os
import torch
import torchaudio.transforms as T
import random
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import math
"데이터 캐글에서 바로 가져오는 코드 추가"

def sound_load(path, target_sr=16000):
    # torchcodec 에러 발생..
    # librosa 보다 sf.read가 빠름
    wav, sr = sf.read(path, dtype="float32")
    wav = torch.from_numpy(wav)

    # shape 정리, soundfile의 경우 보통 (samples, channels) 사용
    if wav.ndim == 1:
        # (N,) → (1, N) 으로 reshape
        wav = wav.unsqueeze(0)
    else:
        # (samples, channels) → (channels, samples)
        if wav.shape[0] > wav.shape[1]:
            wav = wav.transpose(0, 1)  # 디버깅 용이

    # 스트레오 타입을 모노 타입으로 (평균값 이용)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # 리샘플링 기능 추가 (샘플링 레이트 16khz, urbansound8k는 16khz 아닐 수 있음, 추출 혹은 보간으로 추측)
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        wav = resampler(wav)

    return wav, target_sr


def wav_seg(wav):
    curr_len = wav.size(1)
    target_len = 65280
    segments = []
    n = 0
    if curr_len > target_len:
        num_seg = math.ceil(curr_len / target_len)
        stride = (curr_len - target_len) / (num_seg - 1)

        for i in range(num_seg):
            start = int(i * stride)
            segment = wav[:, start: start + target_len]
            segments.append(segment)

    elif curr_len < target_len:
        pad_len = target_len - curr_len
        segments.append(torch.nn.functional.pad(wav, (0, pad_len)))
    else:
        segments.append(wav)

    return segments


def audio_merge(clean, noise):
    target_len = 65280
    n_len = noise.shape[1]
    if n_len < clean.shape[1]:
        noise_seg = noise.repeat(1, (target_len // n_len) + 1)[:, :target_len]
    else:
        start = random.randint(0, n_len - target_len)
        noise_seg = noise[:, start: start + target_len]

    # SNR 기반 노이즈 세기 조절
    snr = random.uniform(-5, 5)
    c_rms = torch.sqrt(torch.mean(clean ** 2) + 1e-8)
    n_rms = torch.sqrt(torch.mean(noise_seg ** 2) + 1e-8)
    noise_scaled = noise_seg * (c_rms / ((10 ** (snr / 20)) * n_rms))
    mixed = clean + noise_scaled

    # 클리핑 방지 및 정규화 (비율 유지)
    max_amp = mixed.abs().max() + 1e-8

    # mixed 볼륨이 1.0을 넘어서 클리핑이 발생할 때만 전체 볼륨을 줄여줌
    if max_amp > 1.0:
        mixed = mixed / max_amp
        clean = clean / max_amp  # clean도 '동일한 값'으로 나누어 SNR 비율 유지!

    return clean, mixed


def extract_stft_features(file_path):
    # STFT 계산 과정
    y_stft = torch.stft(file_path, n_fft=512, hop_length=256, win_length=512, return_complex=True)
    y_stft = y_stft[:, :-1, :]

    # Magnitude 저장
    magnitude = torch.abs(y_stft)
    Log_magnitude = torch.log10(magnitude + 1)

    phase = torch.angle(y_stft)

    return Log_magnitude, phase


base_dir  = Path(r"E:\UNET_DATA")
out_mag_clean   = base_dir / "Processed" / "magnitude_clean"
out_mag_noisy   = base_dir / "Processed" / "magnitude_mixed"
out_phase_clean = base_dir / "Processed" / "phase_clean"
out_phase_noisy = base_dir / "Processed" / "phase_mixed"

for folder in [out_mag_clean, out_mag_noisy, out_phase_clean, out_phase_noisy]:
    os.makedirs(folder, exist_ok=True)

# 원본 파일 가져올 경로 및 파일 추적
clean_files = list((base_dir / "cleanvoice" / "train-clean-100" / "LibriSpeech").rglob("*.flac"))
noise_files = list((base_dir / "noise").rglob("*.wav"))

target_sr = 16000
target_len = 65280
error_count = 0

# 전체 다 돌리려면 아래 코드를 지우기
# clean_files = clean_files[:1]

explosion_factor = 5

for clean_sound in tqdm(clean_files):
    try:
        clean_wav, _ = sound_load(clean_sound, target_sr)
        clean_segments = wav_seg(clean_wav)

        for i, clean_seg in enumerate(clean_segments):
            selected_noises = random.sample(noise_files, explosion_factor)

            for j, noise in enumerate(selected_noises):
                noise_wav, _ = sound_load(noise, target_sr)

                c_seg_processed, m_seg_processed = audio_merge(clean_seg, noise_wav)

                c_mag, c_phase = extract_stft_features(c_seg_processed)
                m_mag, m_phase = extract_stft_features(m_seg_processed)

                file_name = f"{clean_sound.stem}_seg{i}_exp{j}.pt"

                torch.save(c_mag.squeeze(0), out_mag_clean / file_name)
                torch.save(m_mag.squeeze(0), out_mag_noisy / file_name)
                torch.save(c_phase.squeeze(0), out_phase_clean / file_name)
                torch.save(m_phase.squeeze(0), out_phase_noisy / file_name)

    except Exception as e:
        error_count += 1
        print(f"\n[Error] '{clean_sound.name}' 처리 중 오류 발생: {e}")
        continue

print(f"\nProcessing Complete. Total Errors: {error_count}")
