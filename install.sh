#! /bin/bash

set -ex

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
cd ~/ 
wget "https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz" 
tar -xvf mujoco210-linux-x86_64.tar.gz 
mkdir ~/.mujoco 
mv ./mujoco210 ~/.mujoco/mujoco210 
rm mujoco210-linux-x86_64.tar.gz

# Set bashrc
echo "" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin" >> ~/.bashrc
echo "export PATH=\"~/.local/bin:\$PATH\"" >> ~/.bashrc
source ~/.bashrc 

# Create folders
cd ~/CSC4444-Project && mkdir logs ckpt

# Install nvidia-driver
sudo apt install -y nvidia-driver-525

# Start ray head
#ray start --head --port=6380 --num-cpus=8 --num-gpus=1 --memory=$((8*4*1024*1024)) --disable-usage-stats

# ssh into the actor server
#ssh ubuntu@172.31.36.88

# Start ray worker
#ray start --address=172.31.9.112:6380 --num-cpus=16 --num-gpus=0 --memory=$((16*2*1024*1024)) --disable-usage-stats
