# 1. Environment

## 1.1 Create a new virtual environment

```
conda create -n pt36 python=3.6
conda activate pt36
```

## 1.2 Install packages

```
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

cd PLANNER
python -m pip install --editable .
python -m pip install -r requirments.txt
```

# 2. Fine-tune and predict

## 2.1 Download DNABERT

Here are DNABERT models:

> [DNABERT3](https://drive.google.com/file/d/1nVBaIoiJpnwQxiz4dSq6Sv9kBKfXhZuM/view?usp=sharing)
>
> [DNABERT4](https://drive.google.com/file/d/1V7CChcC6KgdJ7Gwdyn73OS6dZR_J-Lrs/view?usp=sharing)
>
> [DNABERT5](https://drive.google.com/file/d/1KMqgXYCzrrYD1qxdyNWnmUYPtrhQqRBM/view?usp=sharing)
>
> [DNABERT6](https://drive.google.com/file/d/1BJjqb5Dl2lNMg2warsFQ0-Xvn1xxfFXC/view?usp=sharing)

If you want to use DNABERT models, please cite the following publication:

Yanrong Ji, Zhihan Zhou, Han Liu, Ramana V Davuluri, DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome, Bioinformatics, 2021;, btab083, https://doi.org/10.1093/bioinformatics/btab083

## 2.2 Data processing

If you are going to fine-tune DNABERT with your own data, please process your data into kmer format. The example is shown in below.

```
sequence	label
TGG GGA GAG AGG GGT	1
CAG AGC GCC CCC CCA	0
ATT TTG TGG GGA GAG	0
```

## 2.3 Fine-tune with pre-trained model and predict

We use DNABERT-3 model as example.

```
python fine-tune.py
```

# 3. Ensemble

```
Python pipeline.py --cell_type A
```



All model we trained are provided at web server [http://planner.unimelb-biotools.cloud.edu.au/](http://planner.unimelb-biotools.cloud.edu.au/.)

