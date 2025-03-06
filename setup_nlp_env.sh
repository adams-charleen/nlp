#!/bin/bash

# Exit if any command fails
set -e

# Navigate to working directory
cd /Users/charleenadams/nlp

echo "### Activating Miniconda base environment ###"
source $HOME/miniconda/bin/activate || true

# Remove old environment if it exists
echo "### Removing old Conda environment (if exists) ###"
conda env remove -n myenv || true

# Download Miniconda for Mac M3 (ARM)
echo "### Downloading Miniconda for ARM ###"
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh

# Install Miniconda silently
echo "### Installing Miniconda ###"
bash Miniconda3-latest-MacOSX-arm64.sh -b -p $HOME/miniconda

# Clean up installer
rm Miniconda3-latest-MacOSX-arm64.sh

# Re-activate base environment
source $HOME/miniconda/bin/activate

# Create new Conda environment with Python 3.10
echo "### Creating Conda environment: myenv ###"
conda create -n myenv python=3.10 -y

# Activate the new environment
conda activate myenv

# Install mamba for faster dependency resolution
echo "### Installing Mamba ###"
conda install -c conda-forge mamba -y

# Install core dependencies via Mamba
echo "### Installing dependencies with Mamba ###"
mamba install -c conda-forge spacy=3.7.5 biopython pytorch pandas numpy -y

# Install remaining dependencies via Pip
echo "### Installing additional dependencies via Pip ###"
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz \
    biopython transformers scikit-learn datasets pandas accelerate

# Verify key packages
echo "### Installed packages ###"
conda list | grep -E "spacy|scispacy|blis|biopython|pytorch|transformers|scikit-learn|datasets|accelerate"

# Test package imports
echo "### Testing package imports ###"
python3 -c "import spacy; nlp = spacy.load('en_core_sci_sm'); print('SciSpaCy loaded successfully')" || { echo "SciSpaCy failed to load"; exit 1; }
python3 -c "import Bio; print('BioPython loaded successfully')" || { echo "BioPython failed to load"; exit 1; }
python3 -c "import transformers; print('Transformers loaded successfully')" || { echo "Transformers failed to load"; exit 1; }
python3 -c "import sklearn; print('Scikit-learn loaded successfully')" || { echo "Scikit-learn failed to load"; exit 1; }
python3 -c "import datasets; print('Datasets loaded successfully')" || { echo "Datasets failed to load"; exit 1; }
python3 -c "import accelerate; print('Accelerate loaded successfully')" || { echo "Accelerate failed to load"; exit 1; }

# Ensure environment is active
conda activate myenv

# Run NLP script in the background with logging
echo "### Running nlp.py in the background ###"
nohup python3 nlp.py > output.log 2>&1 &

# Monitor running processes
echo "### Monitoring process ###"
ps aux | grep nlp.py

echo "### Setup complete! ###"
