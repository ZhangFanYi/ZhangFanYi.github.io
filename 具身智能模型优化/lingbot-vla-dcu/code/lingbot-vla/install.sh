pip install https://github.com/huggingface/lerobot/archive/refs/tags/v0.4.2.tar.gz

git submodule update --init --recursive --remote
pip install -e .
pip install -e ./lingbotvla/models/vla/vision_models/lingbot-depth/ --no-deps
pip install -e ./lingbotvla/models/vla/vision_models/MoGe/
pip install flash-attn==2.8.3 --no-build-isolation