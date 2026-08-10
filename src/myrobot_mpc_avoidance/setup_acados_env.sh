#!/bin/bash
# myrobot_mpc_avoidance 环境配置
# 用法: source setup_acados_env.sh

export ACADOS_INSTALL_DIR="${HOME}/桌面/acados_toolkit/acados"
export ACADOS_SOURCE_DIR=$ACADOS_INSTALL_DIR
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}
export LD_LIBRARY_PATH=$ACADOS_INSTALL_DIR/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$ACADOS_INSTALL_DIR/interfaces/acados_template:$PYTHONPATH
echo "[OK] acados env: ACADOS_INSTALL_DIR=$ACADOS_INSTALL_DIR"
