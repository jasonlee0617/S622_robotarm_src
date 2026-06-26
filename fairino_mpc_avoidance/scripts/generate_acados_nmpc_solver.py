#!/usr/bin/env python3
"""
离线生成 acados C 代码，供 C++ 节点编译使用
运行一次即可，除非修改了 OCP 公式

用法:
    source setup_acados_env.sh               # 先加载环境变量
    python3 scripts/generate_acados_nmpc_solver.py   # 从包根目录运行
"""
import os
import re
import sys
from pathlib import Path

# 确保从包根目录运行，c_generated_code_nmpc/ 生成在正确位置
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_PKG_ROOT)

_DEFAULT_ACADOS_DIR = '/home/robot/桌面/acados_toolkit/acados'
_ACADOS_DIR = os.environ.get('ACADOS_INSTALL_DIR', _DEFAULT_ACADOS_DIR)
_ACADOS_TEMPLATE_DIR = os.path.join(_ACADOS_DIR, 'interfaces', 'acados_template')
if not os.path.isdir(_ACADOS_TEMPLATE_DIR):
    raise RuntimeError(
        f"acados_template not found under ACADOS_INSTALL_DIR={_ACADOS_DIR}. "
        "Run: source setup_acados_env.sh"
    )
sys.path.insert(0, _ACADOS_TEMPLATE_DIR)

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import casadi as ca
import numpy as np

import acados_template

_ACADOS_TEMPLATE_FILE = os.path.abspath(acados_template.__file__)
if not _ACADOS_TEMPLATE_FILE.startswith(os.path.abspath(_ACADOS_TEMPLATE_DIR)):
    raise RuntimeError(
        "Imported acados_template from an unexpected path:\n"
        f"  got: {_ACADOS_TEMPLATE_FILE}\n"
        f"  expected under: {_ACADOS_TEMPLATE_DIR}"
    )

NMPC_N = 15
NMPC_DT = 0.02


def _patch_generated_solver_for_current_acados():
    """Keep checked-in generated C compatible with the local acados C API.

    The project currently builds against an acados tree whose C API requires
    ocp_nlp_out in constraint setters, ocp_nlp_in in output setters, and
    ocp_nlp_out in flat dimension helpers. Some acados_template versions still
    emit the older signatures, so patch the generated file immediately after
    export. The substitutions are idempotent and only touch the generated
    solver wrapper.
    """
    header = Path(_ACADOS_DIR) / 'include' / 'acados_c' / 'ocp_nlp_interface.h'
    if not header.exists():
        return

    header_text = header.read_text(encoding='utf-8', errors='ignore')
    needs_new_signature = (
        'ocp_nlp_in *in, ocp_nlp_out *out, int stage' in header_text
        and 'ocp_nlp_out *out, ocp_nlp_in *in' in header_text
        and 'ocp_nlp_out *out, const char *field' in header_text
    )
    if not needs_new_signature:
        return

    solver_c = Path(_PKG_ROOT) / 'c_generated_code_nmpc' / 'acados_solver_s622arm_nmpc.c'
    text = solver_c.read_text(encoding='utf-8')
    original = text

    if not re.search(r'ocp_nlp_out\s*\*\s*nlp_out\s*=\s*capsule->nlp_out;', text):
        text = text.replace(
            '    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;\n\n    int tmp_int = 0;',
            '    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;\n'
            '    ocp_nlp_out* nlp_out = capsule->nlp_out;\n\n'
            '    int tmp_int = 0;',
        )

    text = re.sub(
        r'ocp_nlp_constraints_model_set\(nlp_config, nlp_dims, nlp_in, (?!nlp_out, )',
        'ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, ',
        text,
    )
    text = re.sub(
        r'ocp_nlp_out_set\(nlp_config, nlp_dims, nlp_out, (?!capsule->nlp_in, |nlp_in, )',
        'ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, capsule->nlp_in, ',
        text,
    )

    text = text.replace(
        '    // 4) create nlp_in\n'
        '    capsule->nlp_in = ocp_nlp_in_create(capsule->nlp_config, capsule->nlp_dims);\n'
        '\n'
        '    // 5) setup functions, nlp_in and default parameters',
        '    // 4) create nlp_in and nlp_out\n'
        '    capsule->nlp_in = ocp_nlp_in_create(capsule->nlp_config, capsule->nlp_dims);\n'
        '    capsule->nlp_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);\n'
        '    capsule->sens_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);\n'
        '\n'
        '    // 5) setup functions, nlp_in and default parameters',
    )
    text = text.replace(
        '    // 7) create and set nlp_out\n'
        '    // 7.1) nlp_out\n'
        '    capsule->nlp_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);\n'
        '    // 7.2) sens_out\n'
        '    capsule->sens_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);\n'
        '    s622arm_nmpc_acados_set_nlp_out(capsule);',
        '    // 7) initialize nlp_out\n'
        '    s622arm_nmpc_acados_set_nlp_out(capsule);',
    )

    text = re.sub(
        r'ocp_nlp_dims_get_total_from_attr\((capsules\[0\]->nlp_solver->config,\s*'
        r'capsules\[0\]->nlp_solver->dims),\s*field\)',
        r'ocp_nlp_dims_get_total_from_attr(\1, capsules[0]->nlp_out, field)',
        text,
    )

    if text != original:
        solver_c.write_text(text, encoding='utf-8')
        print('Patched generated acados solver for current ocp_nlp_interface API.')


def create_model():
    model = AcadosModel()
    model.name = 's622arm_nmpc'

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
    _patch_generated_solver_for_current_acados()

    print("acados C 代码生成完成！")
    print(f"  输出目录: {os.path.join(_PKG_ROOT, 'c_generated_code_nmpc')}")
    print(f"  nx={nx}, nu={nu}, nparam={nparam}, N={N}")


if __name__ == '__main__':
    generate_solver()
