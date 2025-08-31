<div align="center">

### GEMINUS: Dual-aware Global and Scene-Adaptive Mixture-of-Experts for End-to-End Autonomous Driving

<a href="https://www.arxiv.org/abs/2507.14456"><img src='https://img.shields.io/badge/arXiv-GEMINUS-red' alt='Paper PDF'></a>

</div>

## Release List
- [x] Getting Started
- [ ] Data Preprocessing
- [ ] Training
- [ ] Open-loop Evalution
- [ ] Close-loop Evalution

## Getting Started

First, clone this repository and set up the required environment by running the commands below.

```bash
# Step 1: Clone the Repository
git clone [https://github.com/newbrains1/GEMINUS.git](https://github.com/newbrains1/GEMINUS.git)
cd GEMINUS

# Step 2: Create Conda Environment & Activate
conda env create -f environment.yml
conda activate geminus

# Step 3: Install PyTorch
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
## Citation
If you find our repo or our paper useful, please use the following citation:
```bibtex
@article{wan2025geminus,
  title={GEMINUS: Dual-aware Global and Scene-Adaptive Mixture-of-Experts for End-to-End Autonomous Driving},
  author={Wan, Chi and Cui, Yixin and Du, Jiatong and Yang, Shuo and Bai, Yulong and Huang, Yanjun},
  journal={arXiv preprint arXiv:2507.14456},
  year={2025}
}
```

## Acknowledgements
Our code is based on several repositories:
- [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive)
- [Bench2Drive-Zoo](https://github.com/Thinklab-SJTU/Bench2DriveZoo/tree/tcp/admlp)
- [TCP](https://github.com/OpenDriveLab/TCP)
