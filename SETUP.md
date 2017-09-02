# Setup

- clone the repository
- install [anaconda](https://www.anaconda.com/download/)
- run `conda env create -f environment.yml`
- run `./download_data.sh` in the `pretrained` folder to download required files for the Infersent model
- in python, run the following once: `import nltk; nltk.download()`
- download `GoogleNews-vectors-negative300.bin` from `https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit` and put it into `pretrained/word2vec/`
- install `conda install matplotlib` if you want to look at infersent visualizations


## Manual Package Installations

Alternatively to running `conda env create -f environment.yml`, you can setup a Python 2.7 environment in anaconda and manually install:

- NLP
  - `conda install -c conda-forge keras fuzzywuzzy`
  - `conda install -c soumith pytorch torchvision`
  - `conda install scikit-learn nltk gensim theano`
- Server
  - `conda install flask flask-cors`


## Getting CUDA (NVidia) support for PyTorch

Make the following adjustments in `encoders/encoders.py`. Replace

```
infersent = torch.load('encoders/infersent/infersent.allnli.pickle', map_location=lambda storage, loc: storage)
infersent.use_cuda = False
```

with 

```
infersent = torch.load('encoders/infersent/infersent.allnli.pickle')
```

**To install pytorch with CUDA support on OSX 10.12.6, I did the following:**

- clone the [pytorch repo](https://github.com/pytorch/pytorch#from-source)
- install [CUDA](https://developer.nvidia.com/cuda-downloads)
- building for CUDA does no work with apple's latest clang, so I needed to downgrade from Apple LLVM version 8.1.0 to Apple LLVM vresion 8.0.0 (check with `clang --version`) by installing the apple command line tools 8.0 from [this list](https://developer.apple.com/download/more/)
- run `MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py install`
- then, I realized my graphics card didn't have enough memory...
