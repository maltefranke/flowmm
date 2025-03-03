import torch
from tqdm import tqdm

from rfm_docking.se3_dock.so3_utils import calc_rot_vf, rotvec_to_rotmat
from rfm_docking.se3_dock.se3_manifold import SE3ProductManifold


def euler_step(
    odefunc, f_t, v_f_t, rot_t, v_rot_t, t0, dt, manifold: SE3ProductManifold
) -> tuple[torch.Tensor, torch.Tensor]:
    return manifold.expmap(f_t, v_f_t * dt, rot_t, v_rot_t * dt)


def midpoint_step(
    odefunc, f_t, v_f_t, rot_t, v_rot_t, t0, dt, manifold: SE3ProductManifold
):
    half_dt = 0.5 * dt

    f_mid, rot_mid = manifold.expmap(f_t, v_f_t * half_dt, rot_t, v_rot_t * half_dt)
    v_f_mid, v_rot_mid, be_pred = odefunc(t0, f_mid, rot_mid)
    return manifold.expmap(f_t, v_f_mid * dt, rot_t, v_rot_mid * dt)


def rk4_step(odefunc, xt, vt, t0, dt, manifold=None):
    raise NotImplementedError

    k1 = vt
    if manifold is not None:
        raise NotImplementedError
    else:
        k2 = odefunc(t0 + dt / 3, xt + dt * k1 / 3)
        k3 = odefunc(t0 + dt * 2 / 3, xt + dt * (k2 - k1 / 3))
        k4 = odefunc(t0 + dt, xt + dt * (k1 - k2 + k3))
        return xt + (k1 + 3 * (k2 + k3) + k4) * dt * 0.125


def heun_step(
    odefunc, f_t, v_f_t, rot_t, v_rot_t, t0, dt, manifold: SE3ProductManifold
):
    """Heun's method."""  # TODO copilot generated, get actual code from Malte
    k1_f = v_f_t
    k1_rot = v_rot_t

    k2_f, k2_rot, be_pred = odefunc(t0 + dt, f_t + dt * k1_f, rot_t + dt * k1_rot)

    return manifold.expmap(
        f_t, 0.5 * dt * (k1_f + k2_f), rot_t, 0.5 * dt * (k1_rot + k2_rot)
    )


@torch.no_grad()
def get_step_fn(method):
    return {
        "euler": euler_step,
        "midpoint": midpoint_step,
        # "rk4": rk4_step,
        "heun": heun_step,
    }[method]


@torch.no_grad()
def integrate_f_and_rot(
    manifold,
    odefunc,
    f0,
    rot_0,
    t,
    method="euler",
    projx=True,
    local_coords=False,
    pbar=False,
):
    step_fn = get_step_fn(method)

    f_ts = [f0]
    v_f_ts = []

    rot_ts = [rot_0]
    v_rot_ts = []

    t0s = t[:-1]
    if pbar:
        t0s = tqdm(t0s)

    f_t = f0
    rot_t = rot_0
    for t0, t1 in zip(t0s, t[1:]):
        dt = t1 - t0
        v_f_t, v_rot_t, be_pred = odefunc(t0, f_t, rot_t)

        f_t, rot_t = step_fn(
            odefunc,
            f_t,
            v_f_t,
            rot_t,
            v_rot_t,
            t0,
            dt,
            manifold=manifold,
        )
        if projx:
            f_t, rot_t = manifold.projx(f_t, rot_t)

        v_f_ts.append(v_f_t)
        f_ts.append(f_t)

        v_rot_ts.append(v_rot_t)
        rot_ts.append(rot_t)

    """v_f_t, v_rot_t, be_pred = odefunc(t1, f_t, rot_t)

    v_f_ts.append(v_f_t)
    f_ts.append(f_t)

    v_rot_ts.append(v_rot_t)
    rot_ts.append(rot_t)"""
    return (
        torch.stack(f_ts),
        torch.stack(v_f_ts),
        torch.stack(rot_ts),
        torch.stack(v_rot_ts),
    )
