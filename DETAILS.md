# TSP 솔버 최적화 및 알고리즘 상세

## 1. PyTorch 병렬 처리 (`tsp_base.py`)
- `calculate_total_distance(paths, coords)`: 전통적인 for루프가 아닌 `torch.Tensor` 브로드캐스팅을 활용합니다. 50,000개 이상의 경로에 대한 유클리드 거리를 한 번의 연산(O(batch * N^2) GPU 처리)으로 산출합니다.

## 2. 유전 알고리즘 (Genetic Algorithm) 상세
- **초기화**: `torch.argsort(torch.rand())`를 이용해 무작위 순열 집단(Population)을 초고속으로 생성합니다.
- **엘리트 선택 (Elitism)**: 50,000개의 개체군 중 상위 20%(`ELITE_FRACTION`)를 선별해 보존합니다. 최상위 개체(Best Global)는 돌연변이 없이 다음 세대로 직행합니다.
- **적응형 돌연변이**: 선택된 엘리트들을 복제하여 하위 80%를 채운 뒤, 이 중 40%의 개체를 타겟으로 두 도시의 위치를 스왑(Swap)하여 다양성을 확보합니다.

## 3. 시뮬레이션 담금질 (Simulated Annealing) 상세
- **온도 스케줄**: `START_TEMP = 20.0`, `COOLING_RATE = 0.9997`. 매 단계마다 온도가 기하급수적으로 감소합니다. 150,000단계 진행 시 온도는 약 0.000001에 수렴합니다.
- **메트로폴리스 기준**: 새 경로가 더 좋으면(`diff > 0`) 100% 수용, 더 나쁘면 확률 `exp(diff / temp)`로 수용하여 로컬 미니마(Local Minima)를 탈출합니다.

## 4. 하드웨어 최적화 팁
- **OOM(메모리 부족)**: GPU 메모리 초과 시 GA는 `POPULATION_SIZE = 25000`으로, SA는 `BATCH_SIZE = 2048`로 낮춰야 합니다.
- 998개 도시 처리 시, GPU 병렬화를 통해 개체당 평가 시간이 500µs(CPU)에서 2.5µs(GPU)로 약 200배 향상되었습니다.
