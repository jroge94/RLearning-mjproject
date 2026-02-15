#!/usr/bin/env bash
# Supports both Linux and Git Bash (Windows). Run from project root.

set -e

# Project root: directory where this script lives (works in Git Bash and Linux)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR"

# Detect Git Bash / MSYS2 / MinGW on Windows
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "mingw"* ]] || [[ -n "${MSYSTEM:-}" ]]; then
    echo "=== Git Bash / Windows detected ==="

    # Install Python deps (use pip from current Python)
    pip install -r requirements.txt

    # Create project folders in script directory
    mkdir -p logs ckpt imgs

    echo ""
    echo "=== MuJoCo on Windows ==="
    echo "Install MuJoCo manually:"
    echo "  1. Download from https://mujoco.org/download"
    echo "  2. Unzip to e.g. \$USERPROFILE/.mujoco/mujoco210"
    echo "  3. Add MUJOCO_PATH or set LD_LIBRARY_PATH in this shell if needed"
    echo ""
    echo "Project dir: $SCRIPT_DIR"
    echo "Folders created: logs, ckpt, imgs"
    exit 0
fi

# --- Linux path ---
set -x

# Install apt
sudo apt-get update
sudo apt-get -y install software-properties-common ca-certificates
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get -y install libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf \
    build-essential curl gcc-11 pkg-config psmisc unzip \
    python3 python3-pip python-is-python3 wget git vim net-tools

# Install pip
pip install -r requirements.txt

# Install mujoco
cd "$HOME"
wget "https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz"
tar -xvf mujoco210-linux-x86_64.tar.gz
mkdir -p "$HOME/.mujoco"
mv ./mujoco210 "$HOME/.mujoco/mujoco210"
rm -f mujoco210-linux-x86_64.tar.gz

# Set bashrc (works in bash and Git Bash)
{
  echo ""
  echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$HOME/.mujoco/mujoco210/bin"
  echo "export PATH=\"\$HOME/.local/bin:\$PATH\""
} >> "$HOME/.bashrc"
# shellcheck source=/dev/null
source "$HOME/.bashrc" 2>/dev/null || true

# Create folders in project directory (not hardcoded path)
cd "$SCRIPT_DIR"
mkdir -p logs ckpt imgs

# Install nvidia-driver (optional, comment out if not on GPU machine)
# sudo apt install -y nvidia-driver-525

# Start ray head (commented out)
# ray start --head --port=6380 --num-cpus=8 --num-gpus=1 --memory=$((8*4*1024*1024)) --disable-usage-stats

# ssh into the actor server
# ssh ubuntu@172.31.36.88

# Start ray worker
# ray start --address=172.31.9.112:6380 --num-cpus=16 --num-gpus=0 --memory=$((16*2*1024*1024)) --disable-usage-stats
