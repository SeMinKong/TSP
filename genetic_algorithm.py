import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# --- 설정값 ---
POPULATION_SIZE = 50000  
GENERATIONS = 10000
MUTATION_RATE = 0.05
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"사용 중인 장치: {DEVICE}")

df = pd.read_csv('2024_AI_TSP.csv', header=None)
coords = torch.tensor(df.values, dtype=torch.float32).to(DEVICE)
num_cities = len(coords)

population = torch.argsort(torch.rand(POPULATION_SIZE, num_cities, device=DEVICE), dim=1)

def get_total_distances(pop, cities):
    # pop: (Pop_Size, Num_Cities) - 인덱스
    # cities: (Num_Cities, 2) - 좌표
    
    # gather를 이용해 인덱스에 맞는 좌표를 가져옴 -> (Pop_Size, Num_Cities, 2)
    # cities[pop]과 같은 효과지만 PyTorch gather 사용
    # 간단하게 팬시 인덱싱 사용
    ordered_cities = cities[pop] 
    
    # 현재 도시들과 한 칸씩 밀린 도시들(다음 도시) 사이의 거리 계산
    # rolled: [0, 1, 2] -> [1, 2, 0] (다음 도시와 연결 및 마지막-시작 연결)
    rolled_cities = torch.roll(ordered_cities, shifts=-1, dims=1)
    
    # (x1-x2)^2 + (y1-y2)^2
    deltas = ordered_cities - rolled_cities
    distances = (deltas ** 2).sum(dim=2).sqrt() # 유클리드 거리
    
    # 각 개체별 총 거리 합계 (Pop_Size,)
    total_dists = distances.sum(dim=1)
    return total_dists

def crossover(parents):
    # 부모 절반씩 섞기 (단순화된 방식, 정석적인 OX보다 성능은 낮지만 속도는 빠름)
    # TSP 특성상 중복이 없어야 하므로 여기서는 정렬 기반의 트릭을 씁니다.
    # 두 부모의 '순위'를 섞어서 argsort로 다시 인덱스를 뽑는 방식입니다.
    
    pop_len = parents.shape[0]
    
    # 부모 A, B 선택 (무작위 셔플)
    idx_a = torch.randperm(pop_len, device=DEVICE)
    idx_b = torch.randperm(pop_len, device=DEVICE)
    
    parent_a = parents[idx_a]
    parent_b = parents[idx_b]
    
    # 자식 생성: 부모 A의 유전자를 무작위 지점에서 자르고 B를 채우는 방식은 
    # 텐서 연산으로 힘들어서, '난수 키' 기반 병합 정렬 방식을 사용합니다.
    # (이 부분은 GPU 병렬화를 위해 휴리스틱하게 구현됨)
    
    # 단순히 절반을 유지하고 나머지는 채우는 방식
    mask = torch.rand(pop_len, num_cities, device=DEVICE) > 0.5
    
    # 실제로는 유효한 TSP 경로(중복 없는)를 만들기 위해
    # 부모 A를 따르되, B의 순서를 참고하여 정렬하는 방식을 씁니다.
    # 하지만 구현이 복잡하므로, 여기서는 "엘리트만 남기고 변이로 해결"하거나
    # CPU 방식보다 무식하지만 빠른 "무작위 섞기 후 중복 제거"는 어렵습니다.
    
    # [타협안] GPU GA TSP는 구현 난이도가 매우 높으므로, 
    # 여기서는 '돌연변이'와 '엘리트 선택'에 집중하고 교배는 
    # "가장 좋은 놈들 복제 + 변이" 전략으로 갑니다.
    return parents.clone() 

# 5. 돌연변이 (Mutation) - Swap
def mutate(pop, rate):
    # rate 확률로 개체 선택
    num_mutations = int(pop.shape[0] * rate)
    if num_mutations == 0: return pop
    
    # 변이할 개체 인덱스
    target_indices = torch.randint(0, pop.shape[0], (num_mutations,), device=DEVICE)
    
    # 바꿀 두 도시의 위치 (col1, col2)
    col1 = torch.randint(0, num_cities, (num_mutations,), device=DEVICE)
    col2 = torch.randint(0, num_cities, (num_mutations,), device=DEVICE)
    
    # Swap 수행 (Gather/Scatter 대신 팬시 인덱싱 활용)
    # 텐서의 값을 직접 바꿉니다.
    # pop[target_indices, col1] <-> pop[target_indices, col2]
    
    val1 = pop[target_indices, col1]
    val2 = pop[target_indices, col2]
    
    pop[target_indices, col1] = val2
    pop[target_indices, col2] = val1
    
    return pop

# --- 메인 루프 ---
print(f"GPU 가속 시작! (Pop: {POPULATION_SIZE}, Gen: {GENERATIONS})")
start_time = time.time()

for i in range(GENERATIONS):
    # 1. 적합도 계산 (거리)
    dists = get_total_distances(population, coords)
    
    # 2. 정렬 (거리가 짧은 순서대로)
    sorted_indices = torch.argsort(dists)
    population = population[sorted_indices]
    dists = dists[sorted_indices]
    
    # 현 세대 최고 기록 출력
    if (i + 1) % 100 == 0:
        print(f"Gen {i+1}: Best Dist = {dists[0].item():.4f}")
    
    # 3. 다음 세대 생성 (엘리트주의 + 변이 전략)
    # 상위 20%는 그대로 보존 (Elite)
    elite_count = int(POPULATION_SIZE * 0.2)
    elites = population[:elite_count]
    
    # 하위 80%는 상위 20%를 복제한 뒤 돌연변이를 일으킴
    # (TSP 교배 연산인 OX는 GPU 병렬화가 매우 까다로워 변이 위주 전략 사용)
    offsprings = elites.repeat(int(POPULATION_SIZE / elite_count), 1)
    
    # 크기 맞추기 (혹시 나누어 떨어지지 않을 경우)
    offsprings = offsprings[:POPULATION_SIZE]
    
    # 돌연변이 적용 (변이율을 높여서 다양성 확보)
    population = mutate(offsprings, rate=0.4) 
    
    # 0번 엘리트는 변이 없이 무조건 보존 (최적해 분실 방지)
    population[0] = elites[0]

end_time = time.time()
print(f"완료! 소요 시간: {end_time - start_time:.2f}초")

# --- 결과 처리 ---
best_route_tensor = population[0] # 이미 정렬되어 있으므로 0번이 베스트
best_dist = dists[0].item()
best_route = best_route_tensor.cpu().numpy()

print(f"최종 최단 거리: {best_dist:.4f}")

# 0번 도시 맨 앞으로 회전
if 0 in best_route:
    zero_idx = np.where(best_route == 0)[0][0]
    best_route = np.concatenate((best_route[zero_idx:], best_route[:zero_idx]))

# 결과 저장
final_coords = df.values[best_route]
pd.DataFrame(final_coords).to_csv('./solution/solution_GA.csv', header=None, index=False)
print("-> solution_gpu.csv 저장 완료")

# 시각화
plt.figure(figsize=(10, 10))
path_coords = np.vstack([final_coords, final_coords[0]])
plt.plot(path_coords[:, 0], path_coords[:, 1], 'c-', linewidth=0.5, alpha=0.8)
plt.scatter(df.values[:, 0], df.values[:, 1], c='blue', s=5)
plt.scatter(path_coords[0, 0], path_coords[0, 1], c='red', marker='*', s=150, zorder=10)
plt.title(f"GPU Optimized TSP (Dist: {best_dist:.2f})")
plt.savefig('./solution/tsp_GA_result.png')