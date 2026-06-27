#!/bin/bash
# superPOD setup script for Dorabot Minions VLA training
# Run once after git clone on superPOD

set -e

echo "=== Setting up Dorabot Minions VLA on superPOD ==="

# 1. Create virtual environment
python3 -m venv venv --without-pip || python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install box2d-py networkx shapely "protobuf<3.21" cmd2 numpy matplotlib
pip install torch transformers accelerate

echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Collect data:    python src/simulator.py --vla-collect -t 60"
echo "  2. Train:           python src/vla/train.py --data src/data/trajectories/"
echo "  3. Evaluate:        python src/simulator.py --vla -t 10"
