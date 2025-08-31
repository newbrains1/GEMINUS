<div align="center">

# GEMINUS: Dual-aware Global and Scene-Adaptive Mixture-of-Experts for End-to-End Autonomous Driving

</div>

- [√] Getting Started
- [ ] Data Preprocessing
- [ ] Training
- [ ] Open-loop Evalution
- [ ] Close-loop Evalution

## Getting Started

First, clone this repository and set up the required environment by running the commands below.

```bash
# Step 1: Clone the Repository
git clone https://github.com/newbrains1/GEMINUS.git
cd GEMINUS

# Step 2: Create Conda Environment & Activate
conda env create -f environment.yml
conda activate geminus

# Step 3: Install PyTorch
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
