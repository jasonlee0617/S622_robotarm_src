#!/usr/bin/env bash

source ~/venvs/graph_executer/bin/activate
source /opt/ros/humble/setup.bash
source /home/robot/fairino_robotarm/install/setup.bash

cd /home/robot/fairino_robotarm/src/GraphExecuter/graph_executer

PYTHONPATH=/home/robot/fairino_robotarm/src/GraphExecuter/graph_executer/NodeGraphQt:$PYTHONPATH \
NO_ALBUMENTATIONS_UPDATE=1 \
python3 main.py
