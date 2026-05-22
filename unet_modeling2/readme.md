# unet_modeling2 파이프라인 및 고도화 내역

이 폴더(`unet_modeling2`)는 오디오 잡음 제거(Audio Denoising)를 위한 U-Net 모델의 고도화된 학습 스크립트(`01_unet_model.py`)와 최종 테스트 스크립트(`02_test.py`)를 포함하고 있습니다. 이전 모델링(modeling1) 대비 학습 안정성과 평가 방식이 크게 개선되었습니다.

---

## 🚀 주요 파일 설명 및 핵심 기능

### 1. `01_unet_model.py` (고도화된 학습 파이프라인)
기존 학습 코드에 최적화 기법 및 음성 평가 지표를 도입하여 모델의 성능과 학습 효율을 극대화했습니다.

- **AMP (Automatic Mixed Precision) 적용:** `torch.amp.autocast`와 `GradScaler`를 도입하여 VRAM 메모리 사용량을 줄이고 GPU 학습 속도를 크게 향상시켰습니다.
- **고급 학습 스케줄링 및 안정화:** 
  - `CosineAnnealingLR` 스케줄러를 적용하여 학습률(Learning Rate)이 에폭에 따라 부드럽게 감소하도록 설정했습니다.
  - `clip_grad_norm_`(Max Norm 5.0)을 적용해 Gradient Exploding(기울기 폭발)을 방지하고 학습을 안정화했습니다.
- **검증(Val) 단계 PESQ 지표 도입:** 단순 L1 Loss뿐만 아니라 사람의 인지적 음질을 평가하는 **PESQ (Perceptual Evaluation of Speech Quality)** 지표를 `torchmetrics`를 통해 계산합니다.
- **복합 Score 기반 Best Model 저장:** `PESQ - L1 Loss`라는 자체적인 점수(Score)를 계산하여, Loss가 낮으면서도 음질(PESQ)이 가장 높은 시점의 모델을 `modeling2_best_unet_model.pth`로 저장합니다.

### 2. `02_test.py` (최종 모델 테스트 및 평가)
학습이 완료된 최고 성능의 모델을 불러와, 테스트 데이터셋에서 객관적인 오디오 복원 성능을 정밀하게 측정합니다.

- **1D Waveform 완벽 복원:** 모델이 출력한 Maginitude Mask를 원본 Noisy 오디오의 Phase(위상)와 결합하고, `inverse_stft` 함수를 거쳐 실제 사람이 들을 수 있는 1차원 시간 영역의 파형(Waveform)으로 완벽하게 복원합니다.
- **종합 오디오 품질 평가:** 복원된 음성(`pred_waveform`)과 깨끗한 원본 음성(`clean_waveform`)을 비교하여 아래의 3가지 지표를 산출합니다.
  - **L1 Loss:** 스펙트로그램 단위의 절대 오차
  - **PESQ (인지적 음성 품질 평가):** 전반적인 음질 점수 (높을수록 좋음)
  - **STOI (단기 객관적 명료도):** 사람이 음성을 얼마나 명확하게 알아들을 수 있는지에 대한 지표 (높을수록 좋음)
- **재현성 보장:** 글로벌 난수 시드(Seed)를 42로 고정하여 언제 테스트해도 일관된 결과를 얻을 수 있도록 설계되었습니다.

---

## ⚙️ 데이터 흐름 및 파이프라인 (Data Pipeline)

1. **데이터 로드 (`DataLoader`):** 
   - Clean 음성(LibriSpeech)과 Noise(UrbanSound8K)가 합성된 입력 데이터(Noisy)와 정답(Clean) 데이터를 스펙트로그램 형태 `[B, 1, F, T]`로 불러옵니다.
   - 평가를 위해 원래 신호의 Phase 정보도 함께 불러옵니다.

2. **모델 예측 (Forward):** 
   - U-Net 모델이 Noisy 스펙트로그램을 입력받아 **Mask(마스크)**를 예측합니다.
   - `Mask * Noisy` 연산을 통해 노이즈가 제거된 스펙트로그램(`prediction_mag`)을 획득합니다.

3. **손실 및 가중치 업데이트 (Train 단계):** 
   - `L1 Loss` 계산 후 AMP 패키지의 `Scaler`를 통해 역전파(Backpropagation)를 수행하여 가중치를 업데이트합니다.

4. **역변환 및 평가 (Test/Val 단계):** 
   - `inverse_stft`로 1차원 음성 파형을 복구합니다.
   - 평가 모듈(`AudioEvaluator`)을 통해 PESQ, STOI 지표를 추출해 최종적인 잡음 제거 능력을 검증합니다.
