import torch
import torchaudio.transforms as T
import random
import soundfile as sf
import math
import os

# =========================
# 오디오 전처리 과정
# =========================

# 오디오 파일 불러오기
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

# 오디오파일 4초길이에 맞추기
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

# clean파일과 noise 파일 merge
# snr를 랜덤으로 설정한 것 
# train_dataset 용
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
    # 수정됨
    mixed = torch.clamp(mixed, -1.0, 1.0)

    return clean, mixed

# snr을 고정되도록 설정
# eval_dataset용
def audio_merge_eval(clean, noise):
    target_len = 65280
    n_len = noise.shape[1]
    if n_len < clean.shape[1]:
        noise_seg = noise.repeat(1, (target_len // n_len) + 1)[:, :target_len]
    else:
        start = 0
        noise_seg = noise[:, start: start + target_len]

    # SNR 기반 노이즈 세기 조절
    snr = 0
    c_rms = torch.sqrt(torch.mean(clean ** 2) + 1e-8)
    n_rms = torch.sqrt(torch.mean(noise_seg ** 2) + 1e-8)
    noise_scaled = noise_seg * (c_rms / ((10 ** (snr / 20)) * n_rms))
    mixed = clean + noise_scaled

    # 클리핑 방지 및 정규화 (비율 유지)
    # 수정됨
    mixed = torch.clamp(mixed, -1.0, 1.0)

    return clean, mixed


# stft 파일 추출
def extract_stft_features(file_path):
    # STFT 계산 과정
    n_fft = 512
    hop_length = 256
    win_length = 512
    window = torch.hann_window(win_length).to(file_path.device)
    y_stft = torch.stft(file_path, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)
    y_stft = y_stft[:, :-1, :]

    # Magnitude 저장
    magnitude = torch.abs(y_stft)
    Log_magnitude = torch.log10(magnitude + 1)

    phase = torch.angle(y_stft)

    return Log_magnitude, phase



# 오디오 저장 함수 (테스트용)
def save_audio(file_path, audio_data, sample_rate=16000):
    if isinstance(audio_data, torch.Tensor):
        audio_data = audio_data.detach().cpu().numpy()

    if audio_data.ndim == 2:
        if audio_data.shape[0] == 1:
            # 모노(1채널)인 경우: [1, 65280] -> [65280] (1차원으로 쭉 폄)
            audio_data = audio_data.squeeze(0)
        else:
            # 혹시 스테레오(2채널 이상)인 경우: [2, 65280] -> [65280, 2]로 뒤집음
            audio_data = audio_data.transpose()

    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    sf.write(file_path, audio_data, sample_rate)
