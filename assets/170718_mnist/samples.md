## Training on the MNIST dataset

GANs
- DCGAN (sigmoid activation on the last layer)
- MAD-GAN
- WGAN and GoGAN
- BEGAN

Datasets
- MNIST

Comments
- All of the GANs are trained with a fixed LR=1-e5 for 100k iterations.
- Disclaimer: no hyper-parameter search has been done yet.

## Generated samples

#### DCGAN / After 5k, 50k, 99k iterations

<img src="DCGAN_mnist_005.png" width="256"><img src="DCGAN_mnist_050.png" width="256"><img src="DCGAN_mnist_099.png" width="256">

#### MADGAN / After 5k, 50k, 99k iterations

<img src="MADGAN_mnist_005.png" width="256"><img src="MADGAN_mnist_050.png" width="256"><img src="MADGAN_mnist_099.png" width="256">

#### WGAN / After 5k, 50k, 99k iterations

<img src="WGAN_mnist_005.png" width="256"><img src="WGAN_mnist_050.png" width="256"><img src="WGAN_mnist_099.png" width="256">

#### GoGAN / Stage 1 after 5k, 50k iters and stage 2 after 98k iters

<img src="GoGAN_mnist_005_1.png" width="256"><img src="GoGAN_mnist_050_1.png" width="256"><img src="GoGAN_mnist_098_2.png" width="256">

#### BEGAN / After 5k, 50k, 99k iterations

<img src="BEGAN_mnist_005.png" width="256"><img src="BEGAN_mnist_050.png" width="256"><img src="BEGAN_mnist_099.png" width="256">
