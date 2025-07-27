from typing import List, Dict, Any
from pulp import (
    LpProblem, LpMaximize, LpVariable, LpBinary, LpStatus,
    lpSum, value, PULP_CBC_CMD   # CBC 是默认开源求解器
)

def max_slo_attainment(
    L: List[float],          # 工作量数组 L_i，单位 token
    d: List[float],          # 截止时间数组 d_i，单位 s
    speed: float,            # 处理器速率 s，单位 token/s
    solver=None              # 可传入 Gurobi、CPLEX 等商业求解器
) -> Dict[str, Any]:
    N = len(L)
    if len(d) != N:
        raise ValueError("L 与 d 长度必须一致")

    # 处理时间 p_i
    p = [L[i] / speed for i in range(N)]

    # 足够大的 M
    M = sum(p)

    # ----------------- 建模 -----------------
    prob = LpProblem("SLO_Attainment_Maximization", LpMaximize)

    # 决策变量
    y = [LpVariable(f"y_{i}", 0, 1, LpBinary) for i in range(N)]
    C = [LpVariable(f"C_{i}", lowBound=0) for i in range(N)]
    z = [
        [
            None if i == j else LpVariable(f"z_{i}_{j}", 0, 1, LpBinary)
            for j in range(N)
        ]
        for i in range(N)
    ]

    # 目标: 最大化满足 SLO 的任务数量
    prob += (1 / N) * lpSum(y)

    # 约束 1: 排序完备 (两两必有先后)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            prob += z[i][j] + z[j][i] == 1, f"total_order_{i}_{j}"

    # 约束 2: 时间推进 (大-M)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            prob += (
                C[i] >= C[j] + p[i] - M * (1 - z[i][j]),
                f"time_propagation_{i}_{j}"
            )

    # 约束 3: 完成时间下界 (>= 自身处理时长)
    for i in range(N):
        prob += C[i] >= p[i], f"processing_time_lb_{i}"

    # 约束 4: SLO 检查 (y_i==1 ⇒ C_i ≤ d_i)
    for i in range(N):
        prob += C[i] <= d[i] + M * (1 - y[i]), f"slo_check_{i}"

    # ----------------- 求解 -----------------
    if solver is None:
        solver = PULP_CBC_CMD(msg=False)  # 不打印求解日志
    prob.solve(solver)

    status = LpStatus[prob.status]
    attainment = sum(value(v) for v in y) / N

    # 排序得到最终调度顺序
    finish_times = [value(C[i]) for i in range(N)]
    schedule = sorted(range(N), key=lambda idx: finish_times[idx])

    return {
        "status": status,
        "attainment": attainment,
        "schedule": schedule,
        "finish_time": [finish_times[i] for i in schedule],
    }


def sjf_baseline(
    L: List[float],    # 工作量 (tokens)
    d: List[float],    # SLO 截止时间 (s)
    speed: float       # 处理器速率 (tokens/s)
) -> Dict[str, Any]:
    p = [l / speed for l in L]

    schedule = sorted(range(len(p)), key=lambda i: p[i])

    finish_times = []
    t = 0.0
    for i in schedule:
        t += p[i]
        finish_times.append(t)

    ok = sum(fin <= d[i] for i, fin in zip(schedule, finish_times))
    attainment = ok / len(L)

    return {
        "schedule": schedule,
        "finish_times": finish_times,
        "attainment": attainment
    }


def highest_priority_baseline(
    L: List[float],    # 工作量 (tokens)
    d: List[float],    # SLO 截止时间 (s)
    speed: float       # 处理器速率 (tokens/s)
) -> Dict[str, Any]:
    t = 0.0
    queue = list(
        map(lambda i: (i, (L[i] / speed) / (d[i] - t)), range(len(L))))
    schedule = []
    finish_times = []

    while len(queue) > 0:
        queue = list(map(lambda x: (x[0], -1 if x[1] >= 1 else x[1]), queue))
        print(queue)

        queue.sort(key=lambda x: x[1], reverse=True)
        id, _ = queue.pop(0)
        schedule.append(id)
        finish_times.append(t + L[id] / speed)
        t += L[id] / speed
        queue = list(map(lambda x: (x[0], L[x[0]] / (d[x[0]] - t) if x[1] >= 0 else -1), queue))

    ok = sum(fin <= d[i] for i, fin in zip(schedule, finish_times))
    attainment = ok / len(L)

    return {
        "schedule": schedule,
        "finish_times": finish_times,
        "attainment": attainment
    }


# ----------------- 示例 -----------------
if __name__ == "__main__":
    L_example: List[float] = [28831, 26728, 12588, 20187, 14980, 13695, 8058, 9325]  # tokens
    d_example: List[float] = [0.25322466925240406, 5.756611310205642, 6.097069739858171, 7.358464306682947,
                              7.780255650566687, 8.199983512679694, 8.928726554726722, 9.660310186070717]        # seconds
    speed_example = 5000.0

    result = max_slo_attainment(L_example, d_example, speed_example)
    print("MILP | Solver status :", result["status"])
    print("MILP | SLO attainment :", result["attainment"])
    print("MILP | Schedule (idx):", result["schedule"])
    print("MILP | Finish times  :", result["finish_time"])

    # Compare with SJF baseline
    result = sjf_baseline(L_example, d_example, speed_example)
    print("SJF | SLO attainment :", result["attainment"])
    print("SJF | Schedule (idx):", result["schedule"])
    print("SJF | Finish times  :", result["finish_times"])

    # Compare with highest priority baseline
    result = highest_priority_baseline(L_example, d_example, speed_example)
    print("Highest priority | SLO attainment :", result["attainment"])
    print("Highest priority | Schedule (idx):", result["schedule"])
    print("Highest priority | Finish times  :", result["finish_times"])
