#!/bin/bash

EVERCPT=$(pwd)
chmod +x app.py
# Create and activate Conda environment
echo "Creating Conda environment 'evenv' with Python 3.12 and RNAfold..."
conda create -n evenv python=3.12 bioconda::viennarna bioconda::perl-graph -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate evenv

# Install bpRNA manually
echo "Cloning bpRNA..."
git clone https://github.com/hendrixlab/bpRNA
chmod +x bpRNA/bpRNA.pl
rm -rf bpRNA/.git bpRNA/bpRNA_1m

# Download and extract CPC2_standalone
echo "Downloading and extracting CPC2_standalone..."
curl -L https://github.com/gao-lab/CPC2_standalone/archive/refs/tags/v1.0.1.tar.gz -o CPC2
tar -xzf CPC2

# Build libsvm
echo "Building libsvm..."
cd CPC2_standalone-1.0.1/libs/libsvm
tar -xzf libsvm-3.18.tar.gz
cd libsvm-3.18
make clean && make
cd $EVERCPT

# Install Python packages
echo "Installing Python packages from requirements.txt..."
pip install -r requirements.txt

echo "Setup complete"
