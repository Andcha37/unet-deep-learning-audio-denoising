import os
import urllib.request
import tarfile
from pathlib import Path
from tqdm import tqdm

# =======================
# 자동 다운로드 및 압축 해제
# =======================
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_extract(url, download_dir, extracted_folder_name):
    """
    url : 다운로드할 파일의 링크
    download_dir : 다운로드한 파일 저장할 폴더 위치
    extracted_folder_name : 압축 해제 후 생성할 폴더명
    """
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    archive_path = download_dir / Path(url).name
    extracted_path = download_dir / extracted_folder_name

    # 이미 압축 해제된 폴더가 있는지 확인
    if extracted_path.exists():
        print(f"[{extracted_folder_name}] 폴더가 이미 존재하여 다운로드를 건너뜁니다.")
        return
    
    # 압축 파일이 없다면 다운로드
    if not archive_path.exists():
        print(f"다운로드 중...")
        try:
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=Path(url).name) as t:
                urllib.request.urlretrieve(url, filename=archive_path, reporthook=t.update_to)
        except Exception as e:
            print(f"다운로드 실패 ({url}): {e}")
            return
    else:
        print(f"[{archive_path.name}] 압축 파일이 이미 존재하여 압축파일 다운로드를 건너뜁니다.")

    # 압축해제
    print(f"압축 해체 중...")
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=download_dir)
        print(f"[{extracted_folder_name}] 압축 해제를 완료하였습니다. \n")
    except Exception as e:
        print(f"압축 해제 실패: {e}")


# ===============================================================
# librispeech clean 100 , urbansound8k 원본 오디오 데이터 다운로드
# ===============================================================
# 데이터셋 URL
LIBRISPEECH_URL = "http://www.openslr.org/resources/12/train-clean-100.tar.gz"
URBANSOUND_URL = "https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz"

# 현재 폴더 위치를 베이스 디렉토리로 설정
base_dir = Path(os.path.abspath(".."))

print("데이터셋 다운로드 중...")

download_extract(LIBRISPEECH_URL, base_dir, "LibriSpeech")
download_extract(URBANSOUND_URL, base_dir, "UrbanSound8K")

