# TSP

**[English Version](./README.en.md)**

**Genetic Algorithm, GA**과 **Simulated Annealing, SA**을 활용하여 외판원 문제(Traveling Salesman Problem, TSP)를 해결하는 솔버입니다. **PyTorch**의 텐서 연산을 통해 GPU 병렬 처리를 구현함으로써, 수만 개의 경로를 동시에 탐색하여 대규모 도시 데이터셋을 빠르게 처리합니다.

## 주요 특징

- **GPU 가속 병렬 처리**: PyTorch를 사용하여 50,000개 이상의 개체군을 동시에 평가하고 최적화합니다. CPU 대비 약 200배 이상의 성능 향상을 구현했습니다.
- **두 가지 최적화 알고리즘 지원**:
  - **GA**: 상위 20%를 보존하는 엘리티즘 전략과 적응형 돌연변이를 결합하여 넓은 해 공간을 탐색합니다.
  - **SA**: Metropolis Criterion과 점진적 냉각 스케줄을 통해 정밀한 국소 탐색을 수행합니다.
- **모듈화된 아키텍처**: 공통 유틸리티(`tsp_base.py`)를 통해 일관된 데이터 입출력과 거리 계산, 시각화 로직을 분리하여 관리합니다.
- **자동화된 시각화 리포트**: 최단 경로, 시작점, 모든 도시의 분포를 포함한 고해상도 PNG 리포트를 자동으로 생성합니다.

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
기존의 반복문 기반 최적화 대신, 모든 적합도 평가와 변이 로직을 벡터화된 텐서 연산으로 재설계했습니다. 이를 통해 GPU의 병렬 하드웨어를 최대한 활용하여 복잡한 TSP 인스턴스의 수렴 시간을 극적으로 단축했습니다.

### 2. 적응형 진화 전략 및 냉각 스케줄
GA에서는 최적해의 손실을 방지하기 위한 엘리트 주의 전략을, SA에서는 해 공간의 탐험과 활용의 균형을 맞추기 위한 정밀한 냉각 스케줄링을 각각 구현하여 전역 최적해에 근접한 결과를 도출하도록 설계했습니다.

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

>  **더 자세한 정보가 필요하신가요?**
> 상세한 알고리즘 구현 로직, 수학적 수용 기준(Metropolis) 및 GPU 메모리 프로파일링 팁은 [상세 매뉴얼(DETAILS.md)](./DETAILS.md)에서 확인하실 수 있습니다.

---
PyTorch & 메타휴리스틱 알고리즘으로 구축한 프로젝트입니다.
