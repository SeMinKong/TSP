# TSP

**[English Version](./README.en.md)**

유전 알고리즘(Genetic Algorithm, GA)과 담금질 기법(Simulated Annealing, SA)으로 외판원 문제(Traveling Salesman Problem, TSP)를 푸는 솔버입니다. **PyTorch** 텐서 연산으로 수만 개의 경로를 GPU에서 함께 탐색합니다.

## 주요 특징

- **GPU 가속 병렬 처리**: PyTorch로 50,000개 이상의 개체를 동시에 평가합니다. 기존 CPU 코드와 비교했을 때 약 200배 빨랐습니다.
- **두 가지 최적화 알고리즘 지원**:
  - **GA**: 상위 20%를 보존하는 엘리티즘 전략과 적응형 돌연변이를 결합하여 넓은 해 공간을 탐색합니다.
  - **SA**: Metropolis Criterion과 점진적 냉각 스케줄을 통해 정밀한 국소 탐색을 수행합니다.
- **모듈화된 아키텍처**: 공통 유틸리티(`tsp_base.py`)를 통해 일관된 데이터 입출력과 거리 계산, 시각화 로직을 분리하여 관리합니다.
- **결과 시각화**: 최단 경로, 시작점, 도시 분포를 PNG로 저장합니다.

## 기술 스택

- **병렬 연산**: PyTorch (CUDA)
- **수치 해석**: NumPy, Pandas
- **시각화**: Matplotlib
- **언어**: Python 3.8+

## 프로젝트 구조

```text
├── genetic_algorithm.py      # 유전 알고리즘 최적화 로직
├── simulated_annealing.py    # 시뮬레이션 담금질 최적화 로직
├── tsp_base.py               # 공통 유틸리티 (I/O 및 시각화)
├── 2024_AI_TSP.csv           # 샘플 데이터셋 (998개 도시)
└── solution/                 # 최종 결과물(CSV, PNG) 저장소
```

## 핵심 기술 구현 내용

### 1. PyTorch 기반 대규모 병렬 처리
반복문으로 처리하던 적합도 평가와 변이 로직을 벡터화된 텐서 연산으로 바꿨습니다. 여러 경로를 한꺼번에 계산해 GPU 병렬 처리를 사용합니다.

### 2. 적응형 진화 전략 및 냉각 스케줄
GA에서는 상위 20%를 다음 세대로 넘기고 적응형 돌연변이를 적용합니다. SA에서는 Metropolis 기준과 냉각 스케줄로 후보 경로의 수용 여부를 결정합니다.

## 빠른 시작

### 사전 요구사항
- Python 3.8 이상
- [PyTorch](https://pytorch.org/) (성능을 위해 CUDA 버전 권장)

### 설치 및 실행
```bash
git clone <repository-url>
cd TSP
pip install torch numpy pandas matplotlib

# 유전 알고리즘 실행
python genetic_algorithm.py

# 시뮬레이션 담금질 실행
python simulated_annealing.py
```

## 성능 비교
- **GA**: 초기 수렴 속도가 빠르며, 광범위한 영역을 탐색하는 데 유리합니다.
- **SA**: 냉각 단계에서 세밀한 조정을 통해 최종 경로의 품질을 높이는 데 효과적입니다.

알고리즘 구현, Metropolis 수용 기준, GPU 메모리 관련 내용은 [상세 매뉴얼](./DETAILS.md)에 정리했습니다.
