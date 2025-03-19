## CORN: Co-Trained Full- And No-Reference Speech Quality Assessment

<a href='https://doi.org/10.48550/arXiv.2310.09388'><img src='https://img.shields.io/badge/Arxiv-2312.04884-DF826C'></a> 

#### Note: Unofficial re-implementation of CORN.

### 🔨 Installation

Install required Python packages

```
conda create -n corn python=3.9
conda activate corn
pip install -r requirements.txt
```

### 💻 How to use

#### Prepare your data
- Create a data directory on disk and change the data location in **mian.py**.
- **Note**: you should have **a clean audio folder**, **a distorted audio folder**, and **a csv file** with file name information and quality scores (please modify the recognizable columns in your csv file in **main.py**).
- **Note**: The model parameters given are nisqa-corpus training results, which you can choose to call directly.

#### Operation
- **python main.py** 

- **Dataset**: Due to the need of the experiment, we used [NISQA-Corpus](https://github.com/gabrielmittag/NISQA?tab=readme-ov-file) for training, evaluation and testing.
- **Code & Model**: Due to the needs of the experiment, we used NISQA-Corpus for training, evaluation and testing. The framework of corn has been modified from the original text, it is not completely consistent.

### 🪬 Citation

```
@misc{manocha2024corncotrainedfullnoreference,
      title={CORN: Co-Trained Full- And No-Reference Speech Quality Assessment}, 
      author={Pranay Manocha and Donald Williamson and Adam Finkelstein},
      year={2024},
      eprint={2310.09388},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2310.09388}, 
}
```
