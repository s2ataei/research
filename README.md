# About

This repo contains source code for the methods described in: 

[Cascaded Convolutional Neural Networks with Perceptual Loss for Low Dose CT Denoising](https://arxiv.org/abs/2006.14738)

## Usage

DRL contains model and training code for the DRL model using MSE

DRLP contains model and training code for DRL model using Perceptual Loss

LowDose2NormalDose contains model and training code for a supervised image denoising technique based on [Pix2Pix](https://phillipi.github.io/pix2pix/)

`utils.py` contains useful functions for manipulating the [AAPM Low Dose CT Grand Challenge Dataset](https://www.aapm.org/GrandChallenge/LowDoseCT/)

