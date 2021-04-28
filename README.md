# Building Facades to Normal Maps: Adversarial Learning from Single View Images

This repository contains the code for our `Building Facades to Normal Maps – Adversarial Learning from Single View Images` work accepted at [`CRV 2021`](https://www.computerrobotvision.org/).

#### [Paper]() | [Project page](https://mukulkhanna.github.io/building-facade-normal-estimation-crv/) 

![](/assets/preview.png)

## Downloads

Please visit our [project website](https://mukulkhanna.github.io/building-facade-normal-estimation-crv/) for the overview and download links of the custom Synthia dataset with building plane instance annotations and normal maps. The Holicity dataset can be downlaoded from their [website](https://holicity.io
).

## Pre-requisites

- Install all dependencies: `pip install -r requirements.txt`
- Download Synthia dataset, unzip, and place all contents inside `data/synthia` folder.

(Currently the data-loader only supports the custom Synthia dataset.)


## Usage

### Training

```bash
> python train.py -h
usage: train.py [-h] [-e E] [-b [B]] [-l [LR]] [-f LOAD] [-d DATASET]
                [-s SCALE] [-di] [-n NAME] [--resnet]

Train the NormalNet on images and target normal maps.

optional arguments:
  -h, --help            show this help message and exit
  -e E, --epochs E      Number of epochs (default: 5)
  -b [B], --batch-size [B]
                        Batch size (default: 1)
  -l [LR], --learning-rate [LR]
                        LearningFalse rate (default: 0.0001)
  -f LOAD, --load LOAD  Load model from a .pth file (default: False)
  -d DATASET, --dataset DATASET
                        Dataset to be used (default: synthia)
  -s SCALE, --scale SCALE
                        Downscaling factor of the images (default: 0.5)
  -di, --disc           Whether discriminator needs to be trained (default:
                        False)
  -n NAME, --name NAME  Name of experiment (default: None)
  --resnet, -res        Use pre-trained resnet encoder. (default: False)

```
- By default, `--scale` is set to 0.5, so if you wish to obtain better results (but use more memory), set it to 1.
- Add `--resnet` to use the pre-trained resnet encoder.
- Add `--disc` to use the PatchGAN discriminator.


### Prediction

After training your model and saving it to MODEL.pth, you can easily test the output masks on your images via the CLI.

To predict a single image and save it:

`python predict.py -i image.jpg -o output.jpg`

To predict a multiple images and show them without saving them:

`python predict.py -i image1.jpg image2.jpg --viz --no-save`

```shell script
> python predict.py -h
usage: predict.py [-h] [--model FILE] --input INPUT [INPUT ...]
                  [--output INPUT [INPUT ...]] [--viz] [--no-save]
                  [--mask-threshold MASK_THRESHOLD] [--scale SCALE]

Predict masks from input images

optional arguments:
  -h, --help            show this help message and exit
  --model FILE, -m FILE
                        Specify the file in which the model is stored
                        (default: MODEL.pth)
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        filenames of input images (default: None)
  --output INPUT [INPUT ...], -o INPUT [INPUT ...]
                        Filenames of ouput images (default: None)
  --viz, -v             Visualize the images as they are processed (default:
                        False)
  --no-save, -n         Do not save the output masks (default: False)
  --mask-threshold MASK_THRESHOLD, -t MASK_THRESHOLD
                        Minimum probability value to consider a mask pixel
                        white (default: 0.5)
  --scale SCALE, -s SCALE
                        Scale factor for the input images (default: 0.5)
```
You can specify which model file to use with `--model MODEL.pth`.


### Pretrained model
A [pretrained model](https://github.com/milesial/Pytorch-UNet/releases/tag/v1.0) is available for the Carvana dataset. It can also be loaded from torch.hub:

```python
net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana')
```
The training was done with a 100% scale and bilinear upsampling.

## Tensorboard
You can visualize in real time the train and test losses, along with the model predictions with tensorboard:

`tensorboard --logdir=runs`

You can find a sample training run of an experiment with the Synthia dataset on [TensorBoard.dev](https://tensorboard.dev/experiment/d0MFnNlHRYKvq6oeOSp3YA/) (only scalars are supported currently).


## Acknowledgement

This repository has utilized code from the Pytorch-UNet(https://github.com/milesial/Pytorch-UNet) and [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) for the UNet and discriminator implementations.

<style>

code {
  color: pink;
}

</style>