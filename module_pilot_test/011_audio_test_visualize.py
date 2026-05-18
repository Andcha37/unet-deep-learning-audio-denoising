import torch
import os
import random
import matplotlib.pyplot as plt
from pathlib import Path
from preprocess_utils.audio_preprocess import save_audio, sound_load, wav_seg, audio_merge, extract_stft_features

"""
오디오 파일 랜덤으로 불러와서 
데이터 전처리 
stft 시각화
텐서 파일 shape 확인 하는 코드
"""

# 파일 선택 
def get_random_audio(clean_dir, noise_dir):
    clean_files = list(Path(clean_dir).rglob("*.wav"))
    noise_files = list(Path(noise_dir).rglob("*.wav"))

    clean_path = random.choice(clean_files)
    noise_path = random.choice(noise_files)

    return clean_path, noise_path

# 데이터 전처리
def process_to_spectogram(clean_path, noise_path):
    clean_wav, _ = sound_load(clean_path)
    noise_wav, _ = sound_load(noise_path)

    c_seg_processed, m_seg_processed = audio_merge(clean_wav, noise_wav)

    c_mag, _ = extract_stft_features(c_seg_processed)
    m_mag, _ = extract_stft_features(m_seg_processed)

    return c_mag, m_mag, c_seg_processed, noise_wav, m_seg_processed

# 실제 db로 변환
def to_db_spectrogram(log_mag):
    # log_mag = log10(magnitude + 1)
    magnitude = (10 ** log_mag) - 1

    db = 20 * torch.log10(magnitude + 1e-8)

    return db

# 시각화 및 이미지 저장
def plot_spectograms(clean_mag, mixed_mag, output_dir, clean_path, noise_path):
    clean_db = to_db_spectrogram(clean_mag)
    mixed_db = to_db_spectrogram(mixed_mag)

    clean_img = clean_db.squeeze(0).numpy()
    mixed_img = mixed_db.squeeze(0).numpy()

    clean_name = clean_path.stem
    noise_name = noise_path.stem

    plt.figure(figsize=(12, 8))

    plt.subplot(1, 2, 1)
    plt.title("Clean Audio STFT (Label)", fontsize=14)
    plt.imshow(clean_img, aspect='auto', origin='lower', cmap='RdBu_r')
    plt.ylabel("Frequency Bins (0~255)")
    plt.xlabel("Time Frames (0~255)")
    plt.colorbar(format="%+2.0f dB")

    # 오른쪽 그림 (입력: Noisy)
    plt.subplot(1, 2, 2)
    plt.title("Noisy Audio STFT (Input)", fontsize=14)
    plt.imshow(mixed_img, aspect='auto', origin='lower', cmap='RdBu_r')
    plt.xlabel("Time Frames (0~255)")
    plt.colorbar(format="%+2.0f dB")

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    img_save_path = os.path.join(output_dir, f"spectogram_{clean_name}_AND_{noise_name}.png")
    plt.savefig(img_save_path, dpi=300, bbox_inches='tight')

    plt.show()

# 오디오 파일 저장해주기
def save_used_audio(output_dir, clean_audio, noise_audio, mixed_audio, clean_path, noise_path):
    os.makedirs(output_dir, exist_ok=True)

    clean_name = clean_path.stem
    noise_name = noise_path.stem

    clean_save_path = os.path.join(output_dir, f"{clean_name}.wav")
    noise_save_path = os.path.join(output_dir, f"{noise_name}.wav")
    mixed_save_path = os.path.join(output_dir, f"mixed_{clean_name}_and_{noise_name}.wav")

    save_audio(clean_save_path, clean_audio)
    save_audio(noise_save_path, noise_audio)
    save_audio(mixed_save_path, mixed_audio)

# 메인 실행
def main():
    CLEAN_DIR = "./LibriSpeech_Segments"
    NOISE_DIR = "./UrbanSound8K/audio"

    OUTPUT_DIR = "./test_audio_samples"

    try:
        print("오디오 파일 무작위 선택 중...")
        clean_path, noise_path = get_random_audio(CLEAN_DIR, NOISE_DIR)
        print(f"clean 오디오 파일 : {clean_path.name}")
        print(f"noise 오디오 파일 : {noise_path.name}\n")

        print("전처리 및 stft 변환 중...")
        clean_mag, mixed_mag, clean_audio, noise_audio, mixed_audio = process_to_spectogram(clean_path, noise_path)
        print(f"변환 완료! 텐서 크기 : {clean_mag.shape}\n")

        print("오디오 파일 저장 중...")
        save_used_audio(OUTPUT_DIR, clean_audio, noise_audio, mixed_audio, clean_path, noise_path)

        print("스펙트로그램 이미지 생성 중...")
        plot_spectograms(clean_mag, mixed_mag, OUTPUT_DIR, clean_path, noise_path)

    except Exception as e:
        print(f"에러 발생 : {e}")

if __name__ == "__main__":
    main()


