"""SO(3) group transformations."""

import kornia.geometry.conversions as C
import torch
from torch import Tensor
from math import pi as PI


@torch.jit.script
def quat_to_mat(quat_wxyz: Tensor) -> Tensor:
    """Convert scalar first quaternion to rotation matrix.

    Args:
        quat_wxyz: (...,4) Scalar first quaternions.

    Returns:
        (...,3,3) 3D rotation matrices.
    """
    return C.quaternion_to_rotation_matrix(
        quat_wxyz, order=C.QuaternionCoeffOrder.WXYZ
    )


# @torch.jit.script
def mat_to_quat(mat: Tensor) -> Tensor:
    """Convert rotation matrix to scalar first quaternion.

    Args:
        mat: (...,3,3) 3D rotation matrices.

    Returns:
        (...,4) Scalar first quaternions.
    """
    return C.rotation_matrix_to_quaternion(
        mat, order=C.QuaternionCoeffOrder.WXYZ
    )


@torch.jit.script
def quat_to_xyz(
    quat_wxyz: Tensor, singularity_value: float = PI / 2
) -> Tensor:
    """Convert scalar first quaternion to Tait-Bryan angles.

    Reference:
        https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Source_code_2

    Args:
        quat_wxyz: (...,4) Scalar first quaternions.
        singularity_value: Value that's set at the singularities.

    Returns:
        (...,3) The Tait-Bryan angles --- roll, pitch, and yaw.
    """
    qw = quat_wxyz[..., 0]
    qx = quat_wxyz[..., 1]
    qy = quat_wxyz[..., 2]
    qz = quat_wxyz[..., 3]

    # roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    pitch = 2 * (qw * qy - qz * qx)
    is_out_of_range = torch.abs(pitch) >= 1
    pitch[is_out_of_range] = torch.copysign(
        torch.as_tensor(singularity_value), pitch[is_out_of_range]
    )
    pitch[~is_out_of_range] = torch.asin(pitch[~is_out_of_range])

    # yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    xyz = torch.stack([roll, pitch, yaw], dim=-1)
    return xyz


@torch.jit.script
def quat_to_yaw(quat_wxyz: Tensor) -> Tensor:
    """Convert scalar first quaternion to yaw (rotation about vertical axis).

    Reference:
        https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Source_code_2

    Args:
        quat_wxyz: (...,4) Scalar first quaternions.

    Returns:
        (...,) The rotation about the z-axis in radians.
    """
    xyz = quat_to_xyz(quat_wxyz)
    yaw_rad: Tensor = xyz[..., -1]
    return yaw_rad


@torch.jit.script
def xyz_to_quat(xyz_rad: Tensor) -> Tensor:
    """Convert euler angles (xyz - pitch, roll, yaw) to scalar first quaternions.

    Args:
        xyz_rad: (...,3) Tensor of roll, pitch, and yaw in radians.

    Returns:
        (...,4) Scalar first quaternions (wxyz).
    """
    x_rad = xyz_rad[..., 0]
    y_rad = xyz_rad[..., 1]
    z_rad = xyz_rad[..., 2]

    cy = torch.cos(z_rad * 0.5)
    sy = torch.sin(z_rad * 0.5)
    cp = torch.cos(y_rad * 0.5)
    sp = torch.sin(y_rad * 0.5)
    cr = torch.cos(x_rad * 0.5)
    sr = torch.sin(x_rad * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    quat_wxyz = torch.stack([qw, qx, qy, qz], dim=-1)
    return quat_wxyz


@torch.jit.script
def yaw_to_quat(yaw_rad: Tensor) -> Tensor:
    """Convert yaw (rotation about the vertical axis) to scalar first quaternions.

    Args:
        yaw_rad: (...,1) Rotations about the z-axis.

    Returns:
        (...,4) scalar first quaternions (wxyz).
    """
    xyz_rad = torch.zeros_like(yaw_rad)[..., None].repeat_interleave(3, dim=-1)
    xyz_rad[..., -1] = yaw_rad
    quat_wxyz: Tensor = xyz_to_quat(xyz_rad)
    return quat_wxyz
