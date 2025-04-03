# Bidirectional Mamba-Enhanced Speech Tokenization for Spoken Content Retrieval
A fast and efficient speech tokenization approach using bidirectional Mamba for spoken term detection. This repository contains the implementation, datasets, and pre-trained models.


![GitHub stars](https://img.shields.io/github/stars/anupsingh15/BEST-STD?style=social)
![GitHub forks](https://img.shields.io/github/forks/anupsingh15/BEST-STD?style=social)
![License](https://img.shields.io/github/license/anupsingh15/BEST-STD)

## Setup

### Clone the Repository
```sh
git clone https://github.com/anupsingh15/BEST-STD.git
cd BEST-STD
```

### Create a Virtual Environment
```sh
conda create -n best_std anaconda
```

### Install Dependencies
```sh
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install lightning
pip install vector-quantize-pytorch
pip install mamba-ssm
pip install causal-conv1d>=1.4.0
python -m pip install tslearn
pip install librosa
```

### Usage
```sh
Training:

To train the model: Run main.py

Inference:

Creating database, index, and performing retrieval: Run retrieval/std.py

Tokenization demo:

Check demo/word_tokenization.ipynb
```


## Datasets & Pre-trained Models

🔹 **Dataset**: [LibriSpeech Word Alignments](https://github.com/CorentinJ/librispeech-alignments)  
🔹 **Pre-trained Models**: Download from [Google Drive](https://drive.google.com/drive/folders/1Q07VPmitxqGRfdqhJ-dTpV_YsCrMimMC?usp=sharing)

## Citation

If you find our work useful, please cite:
```sh
@inproceedings{singh2025best,
  title={BEST-STD: Bidirectional Mamba-Enhanced Speech Tokenization for Spoken Term Detection},
  author={Singh, Anup and Demuynck, Kris and Arora, Vipul},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```
