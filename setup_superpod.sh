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
pip install torch transformers accelerate peft bitsandbytes

echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Download model:  huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir models/Qwen2.5-7B-Instruct"
echo "  2. Collect data:    python src/simulator.py --vla-collect -t 60"
echo "  3. Train:           python src/vla/train.py --data src/data/trajectories/ --model models/Qwen2.5-7B-Instruct --use-lora --quantize --batch-size 1"
echo "  4. Evaluate:        python src/simulator.py --vla -t 10"
