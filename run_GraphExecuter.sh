#!/usr/bin/env bash

source ~/venvs/graph_executer/bin/activate
source /opt/ros/humble/setup.bash
source $HOME/fairino_robotarm/install/setup.bash

cd $HOME/fairino_robotarm/src/GraphExecuter/graph_executer

PYTHONPATH=$HOME/fairino_robotarm/src/GraphExecuter/graph_executer/NodeGraphQt:$PYTHONPATH \
NO_ALBUMENTATIONS_UPDATE=1 \
python3 main.py
