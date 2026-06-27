#!/bin/bash
set -e
echo "=== Dorabot Minions VLA setup for superPOD ==="

python3 -m venv venv --without-pip || python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install box2d-py networkx shapely "protobuf<3.21" cmd2 numpy matplotlib
pip install torch transformers accelerate peft bitsandbytes

echo ""
echo "=== Setup done. Next steps ==="
echo "1. Download OpenVLA-7B:"
echo "   huggingface-cli download openvla/openvla-7b --include '*.json' '*.safetensors' 'tokenizer*' --local-dir models/openvla-7b/"
echo ""
echo "2. Collect data:"
echo "   python src/simulator.py --vla-collect -t 60"
echo ""
echo "3. Train:"
echo "   python src/vla/train.py --data src/data/trajectories/ --model models/openvla-7b/ --use-lora --quantize"
echo ""
echo "4. Evaluate:"
echo "   python src/simulator.py --vla -t 1"
