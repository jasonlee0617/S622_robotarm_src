#!/usr/bin/env python3
"""
离线生成 acados C 代码，供 C++ 节点编译使用
运行一次即可，除非修改了 OCP 公式

直接执行即可；脚本强制使用包相邻的 mpc_toolbox。
"""
import os
from acados_codegen_common import (
    configure_local_toolbox,
    patch_generated_solver,
    verify_local_imports,
)

_PKG_ROOT, _ACADOS_DIR, _CASADI_DIR, _ACADOS_TEMPLATE_DIR = configure_local_toolbox()

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import casadi as ca
import numpy as np

import acados_template

_ACADOS_TEMPLATE_FILE, _CASADI_FILE = verify_local_imports(
    acados_template, ca, _ACADOS_TEMPLATE_DIR, _CASADI_DIR)

NMPC_N = 15
NMPC_DT = 0.02


def create_model():
    model = AcadosModel()
    model.name = 'fairino_arm_nmpc'

    n = 6  # 关节数
    nx = 2 * n
    nu = n + 1  # [ddq; eps_cbf]

    # 状态和控制
    q = ca.SX.sym('q', n)
    dq = ca.SX.sym('dq', n)
    ddq = ca.SX.sym('ddq', n)
    eps_cbf = ca.SX.sym('eps_cbf')

    x = ca.vertcat(q, dq)
    u = ca.vertcat(ddq, eps_cbf)
    xdot = ca.SX.sym('xdot', nx)

    # 参数
    p_qRef = ca.SX.sym('p_qRef', n)
    p_qdRef = ca.SX.sym('p_qdRef', n)
    p_trackW = ca.SX.sym('p_trackW')
    p_velW = ca.SX.sym('p_velW')
    p_ctrlW = ca.SX.sym('p_ctrlW')
    p_obsGrad = ca.SX.sym('p_obsGrad', n)
    p_obsQuad = ca.SX.sym('p_obsQuad')
    p_obsW = ca.SX.sym('p_obsW')
    p_cbfGradQ = ca.SX.sym('p_cbfGradQ', n)
    p_cbfH = ca.SX.sym('p_cbfH')
    p_cbfVobs = ca.SX.sym('p_cbfVobs')
    p_cbfW = ca.SX.sym('p_cbfW')
    p_cbfGamma = ca.SX.sym('p_cbfGamma')

    p = ca.vertcat(p_qRef, p_qdRef, p_trackW, p_velW, p_ctrlW,
                   p_obsGrad, p_obsQuad, p_obsW,
                   p_cbfGradQ, p_cbfH, p_cbfVobs, p_cbfW, p_cbfGamma)

    # 动力学: q̈ = ddq  (六关节独立 double integrator)
    f_expl = ca.vertcat(dq, ddq)
    f_impl = xdot - f_expl

    model.x = x
    model.u = u
    model.z = ca.SX.sym('z', 0)  # no algebraic variables
    model.xdot = xdot
    model.p = p
    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl

    # 非线性代价
    def smooth_pos(z):
        return 0.5 * (z + ca.sqrt(z * z + 1e-6))

    eq = q - p_qRef
    ev = dq - p_qdRef
    q_track_cost = 4.0 * ca.dot(ca.sin(0.5 * eq), ca.sin(0.5 * eq))
    apf_phi = p_obsQuad + ca.dot(p_obsGrad, eq)
    apf_cost = p_obsW * smooth_pos(apf_phi)**2
    cbf_rate = ca.dot(p_cbfGradQ, dq) + p_cbfVobs
    h_next = p_cbfH + NMPC_DT * cbf_rate + 0.5 * NMPC_DT * NMPC_DT * ca.dot(p_cbfGradQ, ddq)
    cbf_violation_cost = p_cbfW * smooth_pos(-h_next)**2

    model.cost_expr_ext_cost = (
        p_trackW * q_track_cost +
        p_velW * ca.dot(ev, ev) +
        p_ctrlW * ca.dot(ddq, ddq) +
        apf_cost +
        cbf_violation_cost +
        p_cbfW * eps_cbf**2
    )
    model.cost_expr_ext_cost_e = (
        p_trackW * q_track_cost +
        2.0 * p_velW * ca.dot(ev, ev)
    )

    # CBF 约束: ∇h^T q̇ + V_obs + γ h + ε_cbf ≥ 0
    cbf_val = cbf_rate + p_cbfGamma * p_cbfH + eps_cbf
    model.con_h_expr = cbf_val

    return model, n, nx, nu, p.shape[0]


def generate_solver():
    print(f"Using ACADOS_INSTALL_DIR: {_ACADOS_DIR}")
    print(f"Using acados_template: {_ACADOS_TEMPLATE_FILE}")
    print(f"Using CasADi: {_CASADI_FILE}")

    model, n, nx, nu, nparam = create_model()

    N = NMPC_N
    dt = NMPC_DT
    T = N * dt

    ocp = AcadosOcp()
    ocp.model = model
    ocp.dims.N = N
    ocp.dims.nz = 0  # force no algebraic variables
    ocp.code_gen_opts.code_export_directory = os.path.join(_PKG_ROOT, 'c_generated_code_nmpc')

    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'

    # 初始状态约束 — 仅不等式（运行时 C++ 更新 lbx_0/ubx_0 为 [q_now,dq_now]）
    # 不能使用 x0=np.zeros 因为它会同时生成等式约束 idxbxe_0 → 与运行时更新冲突
    ocp.constraints.idxbx_0 = np.arange(nx)
    ocp.constraints.lbx_0 = np.zeros(nx)
    ocp.constraints.ubx_0 = np.zeros(nx)

    # 控制边界
    # ★ MATLAB 一致: MPC 预测时域规划速度限幅（≤ 物理上限）
    dq_max = np.deg2rad([60, 60, 60, 90, 90, 120])
    ddq_max = 2.0 * dq_max
    ocp.constraints.lbu = np.concatenate([-ddq_max, [0.0]])
    ocp.constraints.ubu = np.concatenate([ddq_max, [1e5]])   # MATLAB-compatible relaxed CBF slack upper bound
    ocp.constraints.idxbu = np.arange(nu)

    # 状态边界
    q_min = np.deg2rad([-175.0, -265.0, -162.0, -265.0, -175.0, -175.0])
    q_max = np.deg2rad([175.0, 85.0, 162.0, 85.0, 175.0, 175.0])
    x_min = np.concatenate([q_min, -dq_max])
    x_max = np.concatenate([q_max, dq_max])
    ocp.constraints.lbx = x_min
    ocp.constraints.ubx = x_max
    ocp.constraints.idxbx = np.arange(nx)

    # 终端状态边界（末端预测不可越界，避免激进规划）
    ocp.constraints.idxbx_e = np.arange(nx)
    ocp.constraints.lbx_e = x_min
    ocp.constraints.ubx_e = x_max

    # CBF 约束
    ocp.constraints.lh = np.array([0.0])
    ocp.constraints.uh = np.array([1e5])

    # 求解器选项
    ocp.solver_options.tf = T
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.nlp_solver_max_iter = 20
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.sim_method_num_stages = 2
    ocp.solver_options.sim_method_num_steps = 2

    # 正则化：防止 CBF 约束导致 QP Hessian 奇异引发 ACADOS_MINSTEP
    ocp.solver_options.regularize_method = 'CONVEXIFY'
    ocp.solver_options.reg_epsilon = 1e-6

    ocp.parameter_values = np.zeros(nparam)

    # 生成 C 代码到 c_generated_code_nmpc/，json 存档到 scripts/
    json_path = os.path.join(_PKG_ROOT, 'scripts', 'acados_ocp_fairino_nmpc.json')
    solver = AcadosOcpSolver(ocp, json_file=json_path)
    patch_generated_solver(_PKG_ROOT, _ACADOS_DIR, 'c_generated_code_nmpc', 'fairino_arm_nmpc')

    print("acados C 代码生成完成！")
    print(f"  输出目录: {os.path.join(_PKG_ROOT, 'c_generated_code_nmpc')}")
    print(f"  nx={nx}, nu={nu}, nparam={nparam}, N={N}")


if __name__ == '__main__':
    generate_solver()
