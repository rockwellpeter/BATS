# Bridging Training-Sampling Gap in Conditional Diffusion Models:
**Dynamic Interpolation and Self-Generated Augmentation for Image Restoration**


## Environment Requirements

 We run the code on a computer with `RTX-4090`, and `24G` memory. The code was tested with `python 3.8.13`, `pytorch 1.13.1`, `cudatoolkit 11.7.0`. Install the dependencies via [Anaconda](https://www.anaconda.com/):

```
# create a virtual environment
conda create --name CATS python=3.8.13

# activate environment
conda activate CATS

# install pytorch & cudatoolkit
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

## How to train
We train the model via running:

```
cd Desktop/codes/config/{task}
<br>
python train_ours.py -opt=options/train/ours.yml
```
## How to inference
```
cd Desktop/codes/config/{task}
python test_ours.py -opt=options/test/ours.yml
```


The code of **CATS** is developed based on the code of [improved diffusion](https://github.com/openai/improved-diffusion) and[IR-SDE](https://github.com/Algolzw/image-restoration-sde)


