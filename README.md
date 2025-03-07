# Overview

This script automates the retrieval, classification, and analysis of recent biomedical research using artificial intelligence (AI). Specifically, it searches PubMed for the 10,000 most recent articles (from 2025 onward) related to cardiometabolic diseases, genetic risk factors, and causal mechanisms, then labels and fine-tunes a BioBERT model to classify abstracts based on their relevance to genetic causation. The AI model used here is BioBERT, a transformer-based deep learning model designed for biomedical text mining. BioBERT is a variant of BERT (Bidirectional Encoder Representations from Transformers), a state-of-the-art natural language processing (NLP) model that learns contextual relationships between words in medical literature. The script applies deep learning to process text, identify key biomedical concepts, and classify research abstracts as causally relevant or not. This automated pipeline is valuable for biomedical researchers, clinicians, and data scientists who need to quickly find, analyze, and prioritize research papers without manually sifting through thousands of publications.

## Repository

This repository contains the script **`nlp_10000_most_recent_2025.py`**, which retrieves, processes, and fine-tunes a **BioBERT** model on **10,000** of the most recent **PubMed** abstracts related to **cardiometabolic disorders and genomic causation**. The script uses **biomedical NLP techniques** and **transformers-based deep learning** to classify abstracts based on **causality and genetic associations**.

## Features

- **Automated PubMed Search**: Fetches **10,000+** abstracts using **NCBI's Entrez API**.
- **Natural Language Processing (NLP)**:
  - Uses **SciSpaCy** for biomedical text processing.
  - Extracts terms related to **genetics, cardiometabolic diseases, and causality**.
  - Labels abstracts based on **predefined biomedical categories**.
- **Fine-Tunes BioBERT**:
  - Tokenizes text using `dmis-lab/biobert-base-cased-v1.1`.
  - Implements **imbalanced class handling** with **weighted loss functions**.
  - Trains with **PyTorch** and **Hugging Face Transformers**.
- **Logging and Error Handling**:
  - Logs each step in **`nlp_run_10000_2025.log`**.
  - Implements **retry mechanisms** for API requests.

## Installation and running

Run the following **Bash script** to create a **Miniconda** environment, install dependencies, and run.
Alternately, one could create the Miniconda environment, install dependences, and then run in the background with a log file with: 
`nohup python3 nlp_10000_most_recent_2025.py > nlp_10000_most_recent_2025.log 2>&1 &`

```{bash setup, eval=FALSE, include=TRUE}
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
echo "### Running nlp_10000_most_recent_2025.py in the background ###"
nohup python3 nlp_10000_most_recent_2025.py > nlp_10000_most_recent_2025.log 2>&1 &

# Monitor running processes
echo "### Monitoring process ###"
ps aux | grep nlp.py

echo "### Setup complete! ###"
```
