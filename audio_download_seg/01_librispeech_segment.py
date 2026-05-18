from pathlib import Path
import os
import sys
import soundfile as sf

# ✨ 상위 폴더 경로 추가 (모듈 import 에러 방지)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from preprocess_utils.audio_preprocess import sound_load, wav_seg

BASE_DIR = Path(os.path.abspath(".."))

clean_dir = BASE_DIR / 'LibriSpeech'
save_dir = BASE_DIR / 'LibriSpeech_Segments'

def save_clean_segments(clean_dir, save_dir, target_sr=16000):
    '''
    LibriSpeech의 원본을 dataset에 넣기 전 나누는 함수

    audio_preprocess 모듈에 있는 wav_seg 함수를 이용하고 
    
    idx 넘버를 뒤에 붙여 어떤 파일에서 나누어진건지 표시
    '''
    clean_path = sorted(list(Path(clean_dir).rglob("*.flac")))

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    total_count = 0

    for clean_file in clean_path:
        wav, _ = sound_load(clean_file, target_sr)

        segments = wav_seg(wav)

        stem = clean_file.stem
        
        for idx , seg in enumerate(segments, start=1):
            save_path = save_dir / f"{stem}_{idx:04d}.wav"

            seg_np = seg.squeeze(0).numpy()

            sf.write(save_path, seg_np, target_sr, format='WAV')

            total_count += 1

        print(f"{clean_file.name} -> {len(segments)} segments")

    print(f"\n총 저장된 segment 수: {total_count}")

if __name__ == "__main__":

    save_clean_segments(
        clean_dir=clean_dir,
        save_dir=save_dir
    )
