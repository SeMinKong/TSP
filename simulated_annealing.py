import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# --- [최적화 파라미터 적용] ---
# 도시 1000개, 정규화 좌표(0.xx)에 맞춘 '무겁고 진득한' 설정입니다.
BATCH_SIZE = 4096       # 4096개의 경로를 동시에 탐색 (GPU 메모리 부족시 2048로 줄이세요)
STEPS = 150000          # 충분히 수렴하도록 15만 번 반복
START_TEMP = 20.0       # 좌표 스케일에 맞춰 온도 조정
COOLING_RATE = 0.9997   # 매우 천천히 식힘 (0.9997)
LOG_INTERVAL = 5000     # 5000번 반복할 때마다 로그 출력
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"=== GPU Parallel SA 시작 ===")
print(f"장치: {DEVICE} | 동시 탐색: {BATCH_SIZE}개 | 총 반복: {STEPS}회")
print(f"-------------------------------------------------------------")

# 1. 데이터 로드
try:
    df = pd.read_csv('2024_AI_TSP.csv', header=None)
    coords = torch.tensor(df.values, dtype=torch.float32).to(DEVICE)
    num_cities = len(coords)
except FileNotFoundError:
    print("오류: '2024_AI_TSP.csv' 파일이 없습니다.")
    exit()

# 2. 초기 경로 생성 (랜덤)
current_paths = torch.argsort(torch.rand(BATCH_SIZE, num_cities, device=DEVICE), dim=1)

# 3. 거리 계산 함수
def get_batch_distances(paths, coordinates):
    ordered_coords = coordinates[paths]
    rolled_coords = torch.roll(ordered_coords, shifts=-1, dims=1)
    deltas = ordered_coords - rolled_coords
    distances = (deltas ** 2).sum(dim=2).sqrt().sum(dim=1)
    return distances

# 4. 초기 상태 설정
current_dists = get_batch_distances(current_paths, coords)
best_global_dist = current_dists.min().item()
best_global_path = current_paths[current_dists.argmin()].clone()
start_time = time.time()
temp = START_TEMP

print(f"[초기 상태] 최단 거리: {best_global_dist:.4f}")
print(f"-------------------------------------------------------------")

# --- 메인 루프 ---
for step in range(1, STEPS + 1):
    # (1) 대량 Swap 변이
    new_paths = current_paths.clone()
    
    idx1 = torch.randint(0, num_cities, (BATCH_SIZE,), device=DEVICE)
    idx2 = torch.randint(0, num_cities, (BATCH_SIZE,), device=DEVICE)
    batch_indices = torch.arange(BATCH_SIZE, device=DEVICE)
    
    val1 = new_paths[batch_indices, idx1]
    val2 = new_paths[batch_indices, idx2]
    new_paths[batch_indices, idx1] = val2
    new_paths[batch_indices, idx2] = val1
    
    # (2) 평가 및 수락 (Metropolis)
    new_dists = get_batch_distances(new_paths, coords)
    diff = current_dists - new_dists
    # 온도 T일 때, exp(이득/T) 확률로 수락 (손해여도 가끔 수락)
    probs = torch.exp(diff / temp)
    accept_mask = (diff > 0) | (torch.rand(BATCH_SIZE, device=DEVICE) < probs)
    
    current_paths[accept_mask] = new_paths[accept_mask]
    current_dists[accept_mask] = new_dists[accept_mask]
    
    # (3) 베스트 기록 갱신
    min_dist_batch, min_idx = current_dists.min(dim=0)
    if min_dist_batch.item() < best_global_dist:
        best_global_dist = min_dist_batch.item()
        best_global_path = current_paths[min_idx].clone()
    
    # (4) 온도 감소
    temp *= COOLING_RATE
    
    # --- [진행 상황 출력] ---
    if step % LOG_INTERVAL == 0 or step == STEPS:
        elapsed = time.time() - start_time
        progress = (step / STEPS) * 100
        print(f"[{progress:5.1f}%] 단계: {step:6d} | 온도: {temp:7.4f} | 현재 최단 거리: {best_global_dist:.4f} | 경과: {elapsed:.1f}초")

    if temp < 0.0001:
        print("\n온도가 너무 낮아져 조기 종료합니다.")
        break

# --- 결과 처리 ---
total_time = time.time() - start_time
print(f"-------------------------------------------------------------")
print(f"탐색 완료! 총 소요 시간: {total_time:.2f}초")
print(f"최종 최단 거리: {best_global_dist:.4f}")

# 0번 도시 맨 앞으로 정렬 (Rotation)
final_route = best_global_path.cpu().numpy()
if 0 in final_route:
    zero_idx = np.where(final_route == 0)[0][0]
    ordered_route = np.concatenate((final_route[zero_idx:], final_route[:zero_idx]))
else:
    ordered_route = final_route

# CSV 저장
solution_coords = df.values[ordered_route]
pd.DataFrame(solution_coords).to_csv('solution_SA.csv', header=None, index=False)

# 그래프 저장
plt.figure(figsize=(10, 10))
path_plot = np.vstack([solution_coords, solution_coords[0]])
plt.plot(path_plot[:, 0], path_plot[:, 1], 'g-', linewidth=0.5, alpha=0.7)
plt.scatter(df.values[:, 0], df.values[:, 1], c='gray', s=5, alpha=0.5)
plt.scatter(path_plot[0, 0], path_plot[0, 1], c='red', marker='*', s=200, zorder=10, label='Start')
plt.title(f"Optimized TSP Result (Dist: {best_global_dist:.4f})")
plt.legend()
plt.savefig('tsp_SA_result.png')