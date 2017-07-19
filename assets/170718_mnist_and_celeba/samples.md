## Training on MNIST and CelebA datasets

GANs
- DCGAN (sigmoid activation on the last layer)
- MAD-GAN
- WGAN and GoGAN
- BEGAN

Datasets
- MNIST and CelebA

Comments
- All of the GANs are trained with a fixed LR=1-e5 for 100k iterations.
- Disclaimer: no hyper-parameter search has been done yet.

## Samples

### On MNIST dataset

##### DCGAN / After 5k, 50k, 99k iterations

![DCGAN_MNIST005](DCGAN_mnist_005.png) ![DCGAN_MNIST050](DCGAN_mnist_050.png) ![DCGAN_MNIST099](DCGAN_mnist_099.png)

##### MADGAN / After 5k, 50k, 99k iterations

![MADGAN_MNIST005](MADGAN_mnist_005.png) ![MADGAN_MNIST050](MADGAN_mnist_050.png) ![MADGAN_MNIST099](MADGAN_mnist_099.png)

##### WGAN / After 5k, 50k, 99k iterations

![WGAN_MNIST005](WGAN_mnist_005.png) ![WGAN_MNIST050](WGAN_mnist_050.png) ![WGAN_MNIST099](WGAN_mnist_099.png)

##### GoGAN / Stage 1 after 5k, 50k iters and stage 2 after 98k iters

![GoGAN_MNIST005](GoGAN_mnist_005_1.png) ![GoGAN_MNIST050](GoGAN_mnist_050_1.png) ![GoGAN_MNIST099](GoGAN_mnist_098_2.png)

##### BEGAN / After 5k, 50k, 99k iterations

![BEGAN_MNIST005](BEGAN_mnist_005.png) ![BEGAN_MNIST050](BEGAN_mnist_050.png) ![BEGAN_MNIST099](BEGAN_mnist_099.png)


### On CelebA dataset

##### DCGAN / After 5k, 15k, 20k iterations

- Unfortunately, I've lost latter part of generated images. However, DCGAN wasn't able to generate face-like samples after 20k iterations.

![DCGAN_CELEBA005](DCGAN_celeba_005.png) ![DCGAN_CELEBA050](DCGAN_celeba_015.png) ![DCGAN_CELEBA099](DCGAN_celeba_020.png)

##### MADGAN / After 5k, 50k, 98k iterations

![MADGAN_CELEBA005](MADGAN_celeba_005.png) ![MADGAN_CELEBA050](MADGAN_celeba_050.png) ![MADGAN_CELEBA099](MADGAN_celeba_098.png)

##### WGAN / After 5k, 50k, 99k iterations

![WGAN_CELEBA005](WGAN_celeba_005.png) ![WGAN_CELEBA050](WGAN_celeba_050.png) ![WGAN_CELEBA099](WGAN_celeba_099.png)

##### GoGAN / Stage 1 after 5k, 50k iters and stage 2 after 99k iters

![GoGAN_CELEBA005](GoGAN_celeba_005_1.png) ![GoGAN_CELEBA050](GoGAN_celeba_050_1.png) ![GoGAN_CELEBA099](GoGAN_celeba_099_2.png)

##### BEGAN / After 5k, 50k, 99k iterations

![BEGAN_CELEBA005](BEGAN_celeba_005.png) ![BEGAN_CELEBA050](BEGAN_celeba_050.png) ![BEGAN_CELEBA099](BEGAN_celeba_099.png)
