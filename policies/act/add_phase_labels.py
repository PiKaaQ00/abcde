import numpy as np

PHASE_BASE_APPROACH = 0  # 底盘在动，整体靠近草莓
PHASE_CLEAR          = 1  # 左臂拨叶清障
PHASE_GRASP          = 2  # 右臂对位 + 抓取
PHASE_PLACE          = 3  # 抓完后的拿走/放置

def compute_phase_id(
    actions: np.ndarray,
    left_action_dim: int,
    right_action_dim: int,
    base_action_dim: int,
    th_base_move: float = 0.01,
    th_arm_move: float = 0.01,
    th_diff: float = 0.005,
) -> np.ndarray:
    """
    根据三部分动作（左臂 / 右臂 / 底盘）自动打阶段标签。

    actions: (T, A) numpy 数组
    返回: phase_id: (T,) int 数组，取值为 0/1/2/3
    """
    T, A = actions.shape
    L, R, B = left_action_dim, right_action_dim, base_action_dim
    assert L + R + B == A, "动作维度拆分和总维度不一致"

    left  = actions[:, :L]
    right = actions[:, L:L+R]
    base  = actions[:, L+R:L+R+B]

    # 差分（动作变化幅度）
    left_delta  = np.linalg.norm(left[1:]  - left[:-1],  axis=1)   # (T-1,)
    right_delta = np.linalg.norm(right[1:] - right[:-1], axis=1)   # (T-1,)
    base_delta  = np.linalg.norm(base[1:]  - base[:-1],  axis=1)   # (T-1,)

    phase_id = np.zeros(T, dtype=np.int64)  # 默认全是 BASE_APPROACH

    for t in range(1, T):
        bd = base_delta[t-1]
        ld = left_delta[t-1]
        rd = right_delta[t-1]

        # 1) 底盘在明显运动：BASE_APPROACH
        if bd > th_base_move:
            phase_id[t] = PHASE_BASE_APPROACH
            continue

        # 2) 底盘停了，左臂动得明显多：CLEAR
        if ld > th_arm_move and ld > rd + th_diff:
            phase_id[t] = PHASE_CLEAR
            continue

        # 3) 底盘停了，右臂动得明显多：GRASP
        if rd > th_arm_move and rd > ld + th_diff:
            phase_id[t] = PHASE_GRASP
            continue

        # 4) 其他（底盘停、双臂运动都不明显差异），归为 PLACE
        phase_id[t] = PHASE_PLACE

    # 第 0 帧可以和第 1 帧保持一致，避免孤立
    phase_id[0] = phase_id[1]
    return phase_id
