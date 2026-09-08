import numpy as np

# =============================
# CALIBRATION
# =============================

# Transformation matrices calculated

R_global = np.array([
    [-0.89015,  0.188361,  0.414912],
    [-0.084594, -0.963044, 0.255715],
    [ 0.447746,  0.192526, 0.873188]
])

t_global = np.array([0.679827, 0.710495, 0.518446])

# =============================
# POSITION TRANSFORMS
# =============================

def real_to_mujoco_position(pos_real: np.ndarray) -> np.ndarray:
    """Real (base frame) → MuJoCo world frame"""
    return R_global @ pos_real + t_global


def mujoco_to_real_position(pos_sim: np.ndarray) -> np.ndarray:
    """MuJoCo → Real (inverse transform)"""
    return R_global.T @ (pos_sim - t_global)


# =============================
# DELTA / ACTION TRANSFORMS
# =============================

def mujoco_delta_to_real(delta_sim: np.ndarray) -> np.ndarray:
    """
    Convert delta from MuJoCo frame → real robot frame

    IMPORTANT:
    - No translation applied
    - Only rotation matters
    """
    return R_global.T @ delta_sim


def real_delta_to_mujoco(delta_real: np.ndarray) -> np.ndarray:
    """(Optional) Real → MuJoCo delta"""
    return R_global @ delta_real