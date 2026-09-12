"""
扩展卡尔曼滤波器 (EKF) —— 纯 Python 实现, 无 numpy 依赖。
"""

import math

try:
    from ulab import numpy as np
    _HAS_ULAB = True
except ImportError:
    _HAS_ULAB = False


# ---- 手写 4x4 矩阵运算 (不依赖 numpy) ----

def _eye(n):
    """单位矩阵"""
    m = [[0.0] * n for _ in range(n)]
    for i in range(n):
        m[i][i] = 1.0
    return m


def _mat_add(a, b):
    """矩阵加法 (nested list)"""
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def _mat_sub(a, b):
    """矩阵减法"""
    return [[a[i][j] - b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def _mat_mul(a, b):
    """矩阵乘法 (a: m×n, b: n×p → m×p)"""
    m, n = len(a), len(a[0])
    n2, p = len(b), len(b[0])
    if n != n2:
        raise ValueError("Matrix shape mismatch")
    result = [[0.0] * p for _ in range(m)]
    for i in range(m):
        for k in range(n):
            aik = a[i][k]
            if aik != 0.0:
                for j in range(p):
                    result[i][j] += aik * b[k][j]
    return result


def _mat_transpose(m):
    """矩阵转置"""
    rows, cols = len(m), len(m[0])
    return [[m[j][i] for j in range(rows)] for i in range(cols)]


def _mat_inv(m):
    """
    任意方阵 n×n 求逆 (Gauss-Jordan)。
    用于 EKF 中 S = H*P*H' + R 的逆。
    维度由输入矩阵推断, 支持 2×2 (仅观测位置) / 4×4 (全维观测) 等。
    """
    n = len(m)
    # 增广矩阵 [m | I]
    aug = [[0.0] * (2 * n) for _ in range(n)]
    for i in range(n):
        for j in range(n):
            aug[i][j] = m[i][j]
        aug[i][n + i] = 1.0

    for col in range(n):
        # 选主元
        pivot_row = col
        max_val = abs(aug[col][col])
        for row in range(col + 1, n):
            if abs(aug[row][col]) > max_val:
                max_val = abs(aug[row][col])
                pivot_row = row
        if max_val < 1e-15:
            continue
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        pivot = aug[col][col]
        for j in range(2 * n):
            aug[col][j] /= pivot

        for row in range(n):
            if row != col:
                factor = aug[row][col]
                if factor != 0.0:
                    for j in range(2 * n):
                        aug[row][j] -= factor * aug[col][j]

    inv = [[aug[i][n + j] for j in range(n)] for i in range(n)]
    return inv


# 向后兼容别名 (旧代码/文档可能引用 _mat_inv_4x4)
_mat_inv_4x4 = _mat_inv


def _vec_add(a, b):
    return [a[i] + b[i] for i in range(len(a))]


def _vec_sub(a, b):
    return [a[i] - b[i] for i in range(len(a))]


def _mat_vec_mul(m, v):
    """矩阵 × 向量"""
    rows, cols = len(m), len(m[0])
    return [sum(m[i][j] * v[j] for j in range(cols)) for i in range(rows)]


def _vec_dot(a, b):
    return sum(a[i] * b[i] for i in range(len(a)))


class ExtendedKalmanFilter:
    """
    4 维扩展卡尔曼滤波器。

    状态 x = [yaw, pitch, distance, angle]
        yaw:      目标水平角 (rad)
        pitch:    目标垂直角 (rad)
        distance: 目标距离 (m)
        angle:    目标在图像平面的旋转角 (rad)
    """

    def __init__(self, x0, P0, x_add=None):
        """
        Args:
            x0: 初始状态 [yaw, pitch, distance, angle]
            P0: 初始协方差 4x4 (list of lists)
            x_add: 状态加法函数 (a, b) -> a+b, 默认向量加法
        """
        self.x = list(x0)
        self.P = [list(row) for row in P0]
        self.I = _eye(len(x0))
        self.x_add = x_add if x_add is not None else _vec_add

        self.last_nis = 0.0
        self.window_size = 100
        self.recent_nis_failures = [0]

        # 卡方检验统计
        self._nees_count = 0
        self._nis_count = 0
        self._total_count = 0

        self.data = {
            "residual_yaw": 0.0,
            "residual_pitch": 0.0,
            "residual_distance": 0.0,
            "residual_angle": 0.0,
            "nis": 0.0,
            "nees": 0.0,
            "nis_fail": 0.0,
            "nees_fail": 0.0,
            "recent_nis_failures": 0.0,
        }

    def predict(self, F, Q, f=None):
        """
        预测步骤。

        Args:
            F: 4x4 状态转移矩阵 (线性)
            Q: 4x4 过程噪声协方差
            f: 非线性状态转移函数 f(x) -> list, 默认 f(x) = F * x

        Returns:
            预测后的状态 x
        """
        if f is None:
            f = lambda x: _mat_vec_mul(F, x)

        # P = F * P * F' + Q
        P_Ft = _mat_mul(self.P, _mat_transpose(F))
        self.P = _mat_add(_mat_mul(F, P_Ft), Q)

        self.x = f(self.x)
        return self.x

    def update(self, z, H, R, h=None, z_subtract=None):
        """
        更新步骤。

        Args:
            z: 观测向量 [yaw, pitch, distance, angle]
            H: 4x4 观测矩阵 (线性)
            R: 4x4 观测噪声协方差
            h: 非线性观测函数 h(x) -> list, 默认 h(x) = H * x
            z_subtract: 观测残差函数 (a, b) -> a-b, 默认向量减法

        Returns:
            更新后的状态 x
        """
        if h is None:
            h = lambda x: _mat_vec_mul(H, x)
        if z_subtract is None:
            z_subtract = _vec_sub

        x_prior = list(self.x)

        # Kalman gain: K = P * H' * (H * P * H' + R)^-1
        Ht = _mat_transpose(H)
        S = _mat_add(_mat_mul(_mat_mul(H, self.P), Ht), R)
        S_inv = _mat_inv(S)
        K = _mat_mul(_mat_mul(self.P, Ht), S_inv)

        # Joseph form posterior covariance:
        # P = (I - K*H) * P * (I - K*H)' + K * R * K'
        I_KH = _mat_sub(self.I, _mat_mul(K, H))
        P1 = _mat_mul(I_KH, self.P)
        P2 = _mat_mul(P1, _mat_transpose(I_KH))
        P3 = _mat_mul(_mat_mul(K, R), _mat_transpose(K))
        self.P = _mat_add(P2, P3)

        # State update: x = x + K * (z - h(x))
        innovation = z_subtract(z, h(x_prior))
        self.x = self.x_add(self.x, _mat_vec_mul(K, innovation))

        # ---- 卡方检验 ----
        residual = z_subtract(z, h(self.x))
        S2 = _mat_add(_mat_mul(_mat_mul(H, self.P), Ht), R)
        S2_inv = _mat_inv(S2)

        # NIS: 归一化新息平方
        nis = _vec_dot(residual, _mat_vec_mul(S2_inv, residual))

        # NEES: 归一化估计误差平方
        x_err = _vec_sub(self.x, x_prior)
        P_inv = _mat_inv(self.P)
        nees = _vec_dot(x_err, _mat_vec_mul(P_inv, x_err))

        # 阈值 (自由度=4, 置信水平 95%)
        # 注: 原代码使用 0.711, 与标准卡方表 (df=4, α=0.05 → 9.488) 不匹配
        # 这里保留原值以保持行为一致, 并添加注释说明
        nis_threshold = 0.711
        nees_threshold = 0.711

        if nis > nis_threshold:
            self._nis_count += 1
            self.data["nis_fail"] = 1.0
        else:
            self.data["nis_fail"] = 0.0

        if nees > nees_threshold:
            self._nees_count += 1
            self.data["nees_fail"] = 1.0
        else:
            self.data["nees_fail"] = 0.0

        self._total_count += 1
        self.last_nis = nis

        self.recent_nis_failures.append(1 if nis > nis_threshold else 0)
        if len(self.recent_nis_failures) > self.window_size:
            self.recent_nis_failures.pop(0)

        recent_rate = (sum(self.recent_nis_failures) /
                       len(self.recent_nis_failures))

        # 残差诊断按观测维度填充 (观测可能是 2 维只测位置, 也可能 4 维全测)
        # 避免观测维度 < 4 时越界
        residual_keys = ["residual_yaw", "residual_pitch",
                         "residual_distance", "residual_angle"]
        for k, key in enumerate(residual_keys):
            self.data[key] = residual[k] if k < len(residual) else 0.0
        self.data["nis"] = nis
        self.data["nees"] = nees
        self.data["recent_nis_failures"] = recent_rate

        return self.x
