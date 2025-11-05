# 🎨 생성형 AI & 고급 딥러닝 학습 노트북

> 컴퓨터 비전, 생성형 AI, 자연어 처리 등 고급 딥러닝 기술을 학습하는 Jupyter Notebook입니다.

---

## 📚 프로젝트 개요

이 저장소는 컴퓨터 비전(Computer Vision), 생성형 AI(Generative AI), 자연어 처리(Natural Language Processing) 등 고급 딥러닝 기술을 단계별로 학습할 수 있는 자료를 포함하고 있습니다. OpenCV를 활용한 이미지 처리부터 객체 탐지, 세그멘테이션, GAN, 트랜스포머, ViT까지 다양한 최신 딥러닝 모델을 구현하고 실습합니다.

---

## 📘 주요 학습 내용

### 1. 컴퓨터 비전 (Computer Vision)

컴퓨터 비전은 인공지능의 한 분야로, 컴퓨터가 인간처럼 이미지나 영상을 이해하고 분석할 수 있도록 하는 기술입니다. 객체 검출, 이미지 분류, 얼굴 인식, 장면 이해 등 다양한 작업을 포함하며, 주로 머신러닝과 딥러닝을 활용하여 이미지 속 특징을 추출하고 패턴을 학습합니다.

#### 컴퓨터 비전의 주요 작업

- **이미지 분류**: 사진 속 객체를 카테고리별로 분류
- **객체 탐지 (Object Detection)**: 이미지 내에서 특정 객체의 위치와 종류를 동시에 파악
- **이미지 분할 (Segmentation)**: 이미지의 각 픽셀을 의미 있는 영역으로 구분
- **얼굴 인식**: 얼굴 특징을 추출하여 개인 식별
- **의료 영상 분석**: X-ray, CT, MRI 등 의료 이미지 분석 및 진단 보조

#### 컴퓨터 비전 프레임워크

- **OpenCV**: 이미지 처리 및 컴퓨터 비전 작업을 위한 오픈소스 라이브러리
- **TensorFlow**: Google에서 개발한 프레임워크로 대규모 배포와 성능 최적화에 강점
- **PyTorch**: 직관적인 코드 작성과 디버깅이 용이하여 연구 및 실험에서 많이 사용

### 2. 생성형 AI (Generative AI)

생성형 AI는 기존 데이터를 학습하여 새로운 데이터를 생성하는 AI 기술입니다. 이미지, 텍스트, 음성 등 다양한 형태의 콘텐츠를 생성할 수 있으며, 최근 ChatGPT, DALL-E, Midjourney 등으로 주목받고 있습니다.

#### 생성형 AI의 주요 모델

- **오토인코더 (Autoencoder)**: 데이터의 잠재 표현을 학습하여 재구성 및 차원 축소
- **GAN (Generative Adversarial Network)**: 생성자와 판별자가 경쟁하며 고품질 데이터 생성
- **VAE (Variational Autoencoder)**: 확률적 잠재 공간을 활용한 생성 모델
- **Diffusion Model**: 노이즈를 점진적으로 제거하여 이미지 생성

#### 생성형 AI의 활용 분야

- **이미지 생성**: 새로운 이미지, 스타일 변환, 이미지 복원
- **텍스트 생성**: 자연어 생성, 대화형 AI, 창작 도구
- **데이터 증강**: 학습 데이터 부족 시 데이터셋 확장
- **의료**: 의료 영상 생성, 신약 개발 보조
- **엔터테인먼트**: 게임 콘텐츠 생성, 음악 작곡

### 3. 자연어 처리 (Natural Language Processing, NLP)

자연어 처리는 컴퓨터가 인간의 언어를 이해하고 처리할 수 있도록 하는 AI 기술입니다. 텍스트 분류, 감정 분석, 기계 번역, 챗봇 등 다양한 분야에서 활용됩니다.

#### 자연어 처리의 주요 모델

- **RNN (Recurrent Neural Network)**: 시퀀스 데이터의 시간적 패턴 학습
- **LSTM (Long Short-Term Memory)**: 장기 의존성 문제를 해결한 RNN 변형
- **GRU (Gated Recurrent Unit)**: LSTM의 간소화된 버전
- **Seq2Seq**: 시퀀스를 다른 시퀀스로 변환하는 모델 (기계 번역 등)
- **어텐션 메커니즘**: 중요한 정보에 집중하는 메커니즘
- **트랜스포머 (Transformer)**: 어텐션 메커니즘을 활용한 혁신적 모델
- **사전 학습된 언어 모델**: BERT, GPT 등 대규모 언어 모델

#### 자연어 처리의 주요 작업

- **텍스트 분류**: 뉴스 기사 카테고리 분류, 감정 분석
- **기계 번역**: 언어 간 자동 번역
- **챗봇**: 대화형 AI 시스템 구축
- **문서 요약**: 긴 문서를 짧게 요약
- **질의응답**: 질문에 대한 답변 생성

### 4. 비전 트랜스포머 (Vision Transformer, ViT)

Vision Transformer는 트랜스포머 아키텍처를 이미지 처리에 적용한 모델입니다. 이미지를 패치 단위로 나누어 처리하며, CNN 없이도 뛰어난 성능을 보여주는 혁신적인 모델입니다.

#### ViT의 특징

- **패치 기반 처리**: 이미지를 작은 패치로 분할하여 처리
- **어텐션 메커니즘**: 이미지의 전역적 의존성을 학습
- **CNN 대체 가능**: 합성곱 연산 없이도 우수한 성능 달성

### 5. 비지도 학습 (Unsupervised Learning)

비지도 학습은 레이블이 없는 데이터에서 패턴을 찾아내는 학습 방법입니다. 군집화, 차원 축소, 이상 탐지 등에 활용됩니다.

#### 비지도 학습의 주요 기법

- **군집화 (Clustering)**: 유사한 데이터를 그룹으로 묶기
- **차원 축소**: 고차원 데이터를 저차원으로 변환
- **오토인코더**: 데이터의 잠재 표현 학습

### 6. 딥러닝 학습 프로세스

#### 1. 데이터 준비
- 데이터 수집 및 전처리
- 데이터 증강 (Data Augmentation)
- 학습/검증/테스트 데이터 분할

#### 2. 모델 설계
- 네트워크 구조 정의
- 활성화 함수 선택
- 손실 함수 및 최적화 알고리즘 설정

#### 3. 모델 학습
- 순전파 (Forward Propagation)
- 역전파 (Backpropagation)
- 에포크(Epoch) 반복 학습

#### 4. 모델 평가
- 검증 데이터로 성능 평가
- 과적합(Overfitting) 확인
- 하이퍼파라미터 조정

#### 5. 모델 최적화
- 학습률 조정
- 정규화 기법 적용 (Dropout, Batch Normalization 등)
- 조기 종료 (Early Stopping)

#### 6. 모델 배포
- 모델 저장 및 불러오기
- 추론(Inference) 수행
- 프로덕션 환경 배포

### 7. 생성형 AI를 공부하기 위해 필요한 것들

#### 프로그래밍 언어
- **Python**: 딥러닝의 표준 언어
- **필수 라이브러리**: NumPy, Pandas, Matplotlib, OpenCV

#### 딥러닝 프레임워크
- **PyTorch**: 유연하고 직관적인 인터페이스, 연구 분야에서 널리 사용
- **TensorFlow**: Google에서 개발, 프로덕션 환경에서 널리 사용
- **Keras**: TensorFlow의 고수준 API

#### 컴퓨터 비전 라이브러리
- **OpenCV**: 이미지 처리 및 컴퓨터 비전 작업
- **PIL/Pillow**: 이미지 처리
- **Torchvision**: 이미지 데이터셋 및 전처리 도구

#### 자연어 처리 라이브러리
- **Transformers**: Hugging Face의 사전 학습 모델 라이브러리
- **NLTK**: 자연어 처리 도구
- **spaCy**: 고급 자연어 처리 라이브러리

#### 수학 기초
- **선형대수**: 행렬 연산, 벡터 공간
- **미적분**: 편미분, 체인 룰, 경사 하강법
- **확률과 통계**: 확률 분포, 베이지안 추론

#### 하드웨어
- **GPU**: 대용량 연산을 위한 그래픽 처리 장치 (NVIDIA GPU 권장)
- **클라우드 플랫폼**: Google Colab, AWS, GCP 등

#### 실습 경험
- **Kaggle**: 딥러닝 대회 및 데이터셋
- **Papers with Code**: 최신 논문 및 코드 구현
- **AI Hub**: 한국의 AI 데이터 및 알고리즘 공유 플랫폼
- **오픈소스 프로젝트**: GitHub에서 모델 구현 및 개선

---

## 📁 프로젝트 구조

```
03. ai_study_generativeAI/
├── 01. 컴퓨터 비전.ipynb                        # 컴퓨터 비전 기초 개념
├── 02. Object Detection.ipynb                   # 객체 탐지 모델 학습
├── 03. 이안류 CCTV 데이터셋.ipynb               # CCTV 데이터 분석
├── 04. Segmentation.ipynb                      # 이미지 분할 학습
├── 05. openCV/                                  # OpenCV 실습 코드
│   ├── 1_opencv.py ~ 31_lexsort.py             # OpenCV 기초 실습
│   ├── images/                                  # 실습용 이미지
│   ├── movies/                                  # 실습용 영상
│   └── camera.avi, mix.avi                     # 실습용 영상 파일
├── 06. 차량 파손 데이터셋.ipynb                 # 차량 파손 탐지 프로젝트
├── 07. 비지도 학습.ipynb                        # 비지도 학습 기법
├── 08. 오토인코더.ipynb                        # 오토인코더 구현
├── 09. GAN.ipynb                                # 생성적 적대 신경망 구현
├── 10. 자연어 처리.ipynb                        # 자연어 처리 기초
├── 11. 백터화.ipynb                             # 텍스트 벡터화 기법
├── 12. 신경망 기반의 백터화.ipynb               # 신경망 기반 임베딩
├── 13. RNN.ipynb                                # 순환 신경망 구현
├── 14. LSTM과 GRU.ipynb                        # LSTM과 GRU 모델
├── 15. Seq2Seq.ipynb                            # 시퀀스-투-시퀀스 모델
├── 16. 어텐션 메커니즘.ipynb                    # 어텐션 메커니즘 학습
├── 17. 트랜스포머.ipynb                         # 트랜스포머 모델 구현
├── 18. 사전 학습된 언어 모델(PML).ipynb        # BERT, GPT 등 사전 학습 모델
├── 19. Language 모델의 발전.ipynb              # 언어 모델 발전 과정
├── 20. ViT.ipynb                                # Vision Transformer 구현
├── 21. Sokoto Coventry Fingerprint Dataset.ipynb # 지문 인식 프로젝트
├── 22. 수화 인식 데이터.ipynb                   # 수화 인식 프로젝트
└── README.md                                    # 프로젝트 설명서
```

---

## ⚙️ 실행 환경 설정

### 1️⃣ 가상환경 생성 (선택)

```bash
python -m venv .venv
```

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 2️⃣ 필수 패키지 설치

```bash
# 딥러닝 프레임워크
pip install torch torchvision torchaudio
pip install tensorflow

# 컴퓨터 비전
pip install opencv-python opencv-contrib-python
pip install pillow

# 자연어 처리
pip install transformers
pip install nltk
pip install spacy

# 데이터 처리
pip install numpy pandas matplotlib seaborn
pip install scikit-learn

# 기타 유틸리티
pip install jupyter notebook
pip install tqdm
```

### 3️⃣ Jupyter Notebook 실행

```bash
jupyter notebook
```

---

## 📦 주요 패키지

### 딥러닝 프레임워크
- **PyTorch**: 딥러닝 모델 개발 및 학습
- **TensorFlow**: 딥러닝 모델 개발 및 학습
- **Torchvision**: 이미지 데이터셋 및 전처리 도구

### 컴퓨터 비전
- **OpenCV**: 이미지 처리 및 컴퓨터 비전 작업
- **PIL/Pillow**: 이미지 처리
- **scikit-image**: 과학 이미지 처리

### 자연어 처리
- **Transformers**: Hugging Face의 사전 학습 모델 라이브러리
- **NLTK**: 자연어 처리 도구
- **spaCy**: 고급 자연어 처리 라이브러리

### 데이터 처리
- **NumPy**: 수치 계산 및 배열 연산
- **Pandas**: 데이터 분석 및 조작

### 시각화
- **Matplotlib**: 데이터 시각화
- **Seaborn**: 통계 데이터 시각화

### 기타
- **Scikit-learn**: 머신러닝 유틸리티 및 평가 지표

---

## 💡 핵심 개념

### 생성형 AI (Generative AI)
기존 데이터를 학습하여 새로운 데이터를 생성하는 AI 기술입니다. GAN, 오토인코더, Diffusion Model 등이 대표적인 생성형 AI 모델입니다.

### GAN (Generative Adversarial Network)
생성자(Generator)와 판별자(Discriminator)가 경쟁하며 학습하는 모델입니다. 생성자는 가짜 데이터를 만들고, 판별자는 진짜와 가짜를 구분하려고 하며, 이 과정에서 고품질의 데이터가 생성됩니다.

### 오토인코더 (Autoencoder)
입력 데이터를 압축된 잠재 표현으로 인코딩하고, 이를 다시 원본에 가까운 형태로 디코딩하는 모델입니다. 데이터의 차원 축소, 노이즈 제거, 이상 탐지 등에 활용됩니다.

### 트랜스포머 (Transformer)
어텐션 메커니즘을 활용한 혁신적인 모델 구조입니다. RNN의 순차적 처리 방식의 한계를 극복하고, 병렬 처리를 통해 학습 속도와 성능을 크게 향상시켰습니다. BERT, GPT 등 대규모 언어 모델의 기반이 됩니다.

### 어텐션 메커니즘 (Attention Mechanism)
입력 데이터의 중요한 부분에 집중하는 메커니즘입니다. 시퀀스의 모든 위치를 동시에 고려하여 장거리 의존성을 효과적으로 학습할 수 있게 해줍니다.

### Vision Transformer (ViT)
트랜스포머 아키텍처를 이미지 처리에 적용한 모델입니다. 이미지를 패치 단위로 나누어 처리하며, CNN 없이도 뛰어난 성능을 보여줍니다.

### 객체 탐지 (Object Detection)
이미지 내에서 여러 객체의 위치와 종류를 동시에 파악하는 작업입니다. YOLO, R-CNN, SSD 등이 대표적인 모델입니다.

### 이미지 분할 (Segmentation)
이미지의 각 픽셀을 의미 있는 영역으로 구분하는 작업입니다. Semantic Segmentation과 Instance Segmentation으로 구분됩니다.

### 순전파 (Forward Propagation)
입력 데이터가 네트워크의 각 층을 순차적으로 통과하면서 최종 출력을 생성하는 과정입니다.

### 역전파 (Backpropagation)
학습 과정에서 예측값과 실제값의 차이(오차)를 계산하고, 이를 역방향으로 전파하여 각 가중치와 편향을 업데이트하는 알고리즘입니다.

### 손실 함수 (Loss Function)
모델의 예측값과 실제값 사이의 차이를 측정하는 함수입니다. 회귀 문제에서는 평균 제곱 오차(MSE), 분류 문제에서는 교차 엔트로피(Cross-Entropy)를 주로 사용합니다.

### 과적합 (Overfitting)
모델이 학습 데이터에만 과도하게 맞춰져서 새로운 데이터에 대한 일반화 성능이 떨어지는 현상입니다. 드롭아웃(Dropout), 정규화(Regularization), 조기 종료(Early Stopping) 등의 기법으로 완화할 수 있습니다.

### 전이 학습 (Transfer Learning)
사전 학습된 모델의 가중치를 새로운 작업에 재사용하는 기법입니다. 적은 데이터로도 높은 성능을 달성할 수 있으며, 학습 시간을 단축할 수 있습니다.

---

## 🔗 참고 자료

### 컴퓨터 비전
- [Papers with Code](https://paperswithcode.com/) - 최신 논문 및 코드 구현
- [COCO Dataset](https://cocodataset.org/) - 객체 탐지 및 분할 데이터셋
- [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html) - 대규모 이미지 데이터셋
- [OpenCV 공식 문서](https://docs.opencv.org/)

### 자연어 처리
- [Hugging Face](https://huggingface.co/) - 사전 학습 모델 및 라이브러리
- [Transformers 공식 문서](https://huggingface.co/docs/transformers/)
- [Papers with Code - NLP](https://paperswithcode.com/task/natural-language-processing)

### 생성형 AI
- [Papers with Code - Generative Models](https://paperswithcode.com/task/generative-modeling)
- [GAN 논문 (Ian Goodfellow)](https://arxiv.org/abs/1406.2661)

### 딥러닝 프레임워크
- [PyTorch 공식 문서](https://pytorch.org/docs/stable/index.html)
- [TensorFlow 공식 문서](https://www.tensorflow.org/api_docs)
- [PyTorch 튜토리얼](https://pytorch.org/tutorials/)
- [TensorFlow 튜토리얼](https://www.tensorflow.org/tutorials)

### 데이터 플랫폼
- [Kaggle](https://www.kaggle.com/) - 머신러닝 대회 및 데이터셋
- [AI Hub](https://www.aihub.or.kr/) - 한국의 AI 데이터 및 알고리즘 공유 플랫폼
- [Dacon](https://dacon.io/) - 한국의 데이터 분석 대회 플랫폼

---

## 📝 라이선스

이 프로젝트는 학습 목적으로 작성되었습니다.
