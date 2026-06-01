import numpy as np
from dataclasses import dataclass


@dataclass
class MPC2DConfig:
    ts: float = 0.005
    horizon: int = 32

    # end-effector velocity first-order lag
    tau: float = 0.04

    # explicit pure input delay: delay_steps * ts
    input_delay_steps: int = 5

    # weights
    q_e: float = 120.0
    q_v: float = 4.0
    q_terminal: float = 160.0
    r_u: float = 1.0
    r_du: float = 25.0

    # constraints
    u_max: float = 0.20
    du_max: float = 0.0045
    norm_clip: float = 0.20

    # simple projected-gradient solver
    max_iters: int = 20
    grad_step: float = 0.06
    reg_eps: float = 1e-8


class AxisDelayAwareMPC:
    """
    1D linear MPC with explicit pure input delay.

    state:
        x = [e, v_ee, d0, d1, ..., d_{D-1}]^T

    where:
        e      : target position error
        v_ee   : measured end-effector axis velocity
        d_i    : command values still "in flight" in the delay pipeline
    """

    def __init__(self, cfg: MPC2DConfig):
        self.cfg = cfg
        self.delay_steps = max(0, int(cfg.input_delay_steps))
        self.nx = 2 + self.delay_steps

        self.u_prev = 0.0
        self.u_seq = np.zeros(cfg.horizon, dtype=float)
        self.delay_buf = np.zeros(self.delay_steps, dtype=float)

        self._build_model()
        self._build_prediction_cache()

    def _build_model(self):
        ts = float(self.cfg.ts)
        tau = max(float(self.cfg.tau), 1e-4)
        a = float(np.exp(-ts / tau))
        b = 1.0 - a

        nx = self.nx
        D = self.delay_steps

        A = np.zeros((nx, nx), dtype=float)
        B = np.zeros((nx, 1), dtype=float)
        E = np.zeros((nx, 1), dtype=float)

        # e(k+1) = e(k) - ts * v_ee(k) + ts * v_ref(k)
        A[0, 0] = 1.0
        A[0, 1] = -ts
        E[0, 0] = ts

        # v(k+1)
        A[1, 1] = a
        if D > 0:
            A[1, 2] = b
        else:
            B[1, 0] = b

        # delay pipeline
        if D > 0:
            for i in range(D - 1):
                A[2 + i, 2 + i + 1] = 1.0
            B[2 + D - 1, 0] = 1.0

        self.A = A
        self.B = B
        self.E = E

    def _build_prediction_cache(self):
        """
        Precompute:
            X = M x0 + S U + G W
        where X stacks [x1, x2, ..., xN]
        """
        N = int(self.cfg.horizon)
        nx = self.nx

        M = np.zeros((nx * N, nx), dtype=float)
        S = np.zeros((nx * N, N), dtype=float)
        G = np.zeros((nx * N, N), dtype=float)

        A_pows = [np.eye(nx, dtype=float)]
        for _ in range(1, N + 1):
            A_pows.append(self.A @ A_pows[-1])

        for i in range(N):
            M[i * nx:(i + 1) * nx, :] = A_pows[i + 1]
            for j in range(i + 1):
                A_ij = A_pows[i - j]
                S[i * nx:(i + 1) * nx, j] = (A_ij @ self.B).reshape(-1)
                G[i * nx:(i + 1) * nx, j] = (A_ij @ self.E).reshape(-1)

        self.M = M
        self.S = S
        self.G = G

        Tdu = np.zeros((N, N), dtype=float)
        Tdu[0, 0] = 1.0
        for i in range(1, N):
            Tdu[i, i] = 1.0
            Tdu[i, i - 1] = -1.0
        self.Tdu = Tdu

        Q = np.zeros((nx, nx), dtype=float)
        Q[0, 0] = float(self.cfg.q_e)
        Q[1, 1] = float(self.cfg.q_v)

        QN = np.zeros((nx, nx), dtype=float)
        QN[0, 0] = float(self.cfg.q_terminal)
        QN[1, 1] = float(self.cfg.q_v)

        Qbar = np.zeros((nx * N, nx * N), dtype=float)
        for i in range(N):
            blk = QN if i == (N - 1) else Q
            Qbar[i * nx:(i + 1) * nx, i * nx:(i + 1) * nx] = blk
        self.Qbar = Qbar

        self.Rbar = np.eye(N, dtype=float) * float(self.cfg.r_u)

    def reset(self):
        self.u_prev = 0.0
        self.u_seq[:] = 0.0
        if self.delay_steps > 0:
            self.delay_buf[:] = 0.0

    def _build_initial_state(self, e0: float, v_ee0: float):
        x0 = np.zeros(self.nx, dtype=float)
        x0[0] = float(e0)
        x0[1] = float(v_ee0)
        if self.delay_steps > 0:
            x0[2:] = self.delay_buf
        return x0

    def solve(self, e0: float, v_ee0: float, v_ref):
        N = int(self.cfg.horizon)
        nx = self.nx

        x0 = self._build_initial_state(e0, v_ee0)

        if np.isscalar(v_ref):
            Vref = np.ones(N, dtype=float) * float(v_ref)
        else:
            Vref = np.asarray(v_ref, dtype=float).reshape(-1)
            if Vref.size < N:
                Vref = np.pad(Vref, (0, N - Vref.size), mode="edge")
            elif Vref.size > N:
                Vref = Vref[:N]

        Xref = np.zeros(nx * N, dtype=float)
        for i in range(N):
            Xref[nx * i + 0] = 0.0
            Xref[nx * i + 1] = float(Vref[i])

        Uref = np.clip(Vref, -self.cfg.u_max, self.cfg.u_max)

        b_prev = np.zeros(N, dtype=float)
        b_prev[0] = self.u_prev

        Xbias = self.M @ x0 + self.G @ Vref - Xref

        H = (
            self.S.T @ self.Qbar @ self.S
            + self.Rbar
            + float(self.cfg.r_du) * (self.Tdu.T @ self.Tdu)
            + float(self.cfg.reg_eps) * np.eye(N, dtype=float)
        )
        f = (
            self.S.T @ self.Qbar @ Xbias
            - self.Rbar @ Uref
            - float(self.cfg.r_du) * (self.Tdu.T @ b_prev)
        )

        U = self.u_seq.copy()

        for _ in range(int(self.cfg.max_iters)):
            grad = H @ U + f
            U = U - float(self.cfg.grad_step) * grad
            U = np.clip(U, -self.cfg.u_max, self.cfg.u_max)

            dU = self.Tdu @ U - b_prev
            dU = np.clip(dU, -self.cfg.du_max, self.cfg.du_max)

            U_new = np.zeros_like(U)
            U_new[0] = self.u_prev + dU[0]
            for i in range(1, N):
                U_new[i] = U_new[i - 1] + dU[i]
            U = np.clip(U_new, -self.cfg.u_max, self.cfg.u_max)

        u0 = float(U[0])
        self.u_seq = U.copy()
        self.u_prev = u0

        if self.delay_steps > 0:
            if self.delay_steps > 1:
                self.delay_buf[:-1] = self.delay_buf[1:]
            self.delay_buf[-1] = u0

        debug = {
            "x0": x0.copy(),
            "v_ref_preview": Vref.copy(),
            "u0": u0,
            "u_seq": U.copy(),
            "delay_steps": int(self.delay_steps),
            "delay_buffer": self.delay_buf.copy(),
        }
        return u0, debug


class MPCController2D:
    def __init__(self, cfg: MPC2DConfig):
        self.cfg = cfg
        self.ctrl_x = AxisDelayAwareMPC(cfg)
        self.ctrl_y = AxisDelayAwareMPC(cfg)

    def reset(self):
        self.ctrl_x.reset()
        self.ctrl_y.reset()

    def _clip_norm(self, ux: float, uy: float):
        u = np.array([ux, uy], dtype=float)
        n = float(np.linalg.norm(u))
        if n <= self.cfg.norm_clip or n < 1e-9:
            return float(u[0]), float(u[1])
        u *= (self.cfg.norm_clip / n)
        return float(u[0]), float(u[1])

    def step(self, e_xy, v_ref_xy, v_ee_xy):
        """
        v_ref_xy can be:
            - shape (2,)     : single-step reference velocity
            - shape (N, 2)   : horizon preview sequence
        """
        e_xy = np.asarray(e_xy, dtype=float).reshape(2,)
        v_ee_xy = np.asarray(v_ee_xy, dtype=float).reshape(2,)
        v_ref_xy = np.asarray(v_ref_xy, dtype=float)

        if v_ref_xy.ndim == 1:
            if v_ref_xy.size != 2:
                raise ValueError(f"Expected v_ref_xy shape (2,), got {v_ref_xy.shape}")
            vref_x = float(v_ref_xy[0])
            vref_y = float(v_ref_xy[1])
            v_ref_debug = v_ref_xy.copy()
        elif v_ref_xy.ndim == 2:
            if v_ref_xy.shape[1] != 2:
                raise ValueError(f"Expected v_ref_xy shape (N,2), got {v_ref_xy.shape}")
            vref_x = v_ref_xy[:, 0].copy()
            vref_y = v_ref_xy[:, 1].copy()
            v_ref_debug = v_ref_xy.copy()
        else:
            raise ValueError(f"Unsupported v_ref_xy shape: {v_ref_xy.shape}")

        ux, dbg_x = self.ctrl_x.solve(
            e0=float(e_xy[0]),
            v_ee0=float(v_ee_xy[0]),
            v_ref=vref_x,
        )
        uy, dbg_y = self.ctrl_y.solve(
            e0=float(e_xy[1]),
            v_ee0=float(v_ee_xy[1]),
            v_ref=vref_y,
        )

        ux, uy = self._clip_norm(ux, uy)

        debug = {
            "e_xy": e_xy.copy(),
            "v_ref_xy": v_ref_debug,
            "v_ee_xy": v_ee_xy.copy(),
            "u_xy": np.array([ux, uy], dtype=float),
            "x_axis": dbg_x,
            "y_axis": dbg_y,
        }
        return float(ux), float(uy), debug