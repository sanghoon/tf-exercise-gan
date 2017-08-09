## Training on the CelebA dataset

GANs
- DCGAN (sigmoid activation on the last layer)
- MAD-GAN
- WGAN and GoGAN
- BEGAN

Datasets
- CelebA (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

Comments
- All of the GANs are trained with a fixed LR=1-e5 for 100k iterations.
- Disclaimer: no hyper-parameter search has been done yet.


### Generated samples

#### DCGAN

After 4k, 10k, 50k iterations

<img src="fig_DCGAN_gen_0004.png" width="256"><img src="fig_DCGAN_gen_0010.png" width="256"><img src="fig_DCGAN_gen_0050.png" width="256">

Samples from the last iterations

<img src="fig_DCGAN_gen_0092.png" width="256"><img src="fig_DCGAN_gen_0094.png" width="256"><img src="fig_DCGAN_gen_0096.png" width="256">

<br/>

#### MADGAN

After 4k, 10k, 50k iterations

<img src="fig_MADGAN_gen_0004.png" width="256"><img src="fig_MADGAN_gen_0010.png" width="256"><img src="fig_MADGAN_gen_0050.png" width="256">

Samples from the last iterations

<img src="fig_MADGAN_gen_0092.png" width="256"><img src="fig_MADGAN_gen_0094.png" width="256"><img src="fig_MADGAN_gen_0096.png" width="256">

<br/>

#### WGAN

After 4k, 10k, 50k iterations

<img src="fig_WGAN_gen_0004.png" width="256"><img src="fig_WGAN_gen_0010.png" width="256"><img src="fig_WGAN_gen_0050.png" width="256">

Samples from the last iterations

<img src="fig_WGAN_gen_0092.png" width="256"><img src="fig_WGAN_gen_0094.png" width="256"><img src="fig_WGAN_gen_0096.png" width="256">

<br/>

#### GoGAN

Stage 1 samples after 4k, 10k, 50k iterations

<img src="fig_GGAN_1st_0004.png" width="256"><img src="fig_GGAN_1st_0010.png" width="256"><img src="fig_GGAN_1st_0050.png" width="256">

Stage 2 samples from the last iterations

<img src="fig_GGAN_2nd_0092.png" width="256"><img src="fig_GGAN_2nd_0094.png" width="256"><img src="fig_GGAN_2nd_0096.png" width="256">

<br/>

#### BEGAN

After 4k, 10k, 50k iterations

<img src="fig_BEGAN_gen_0004.png" width="256"><img src="fig_BEGAN_gen_0010.png" width="256"><img src="fig_BEGAN_gen_0050.png" width="256">

Samples from the last iterations

<img src="fig_BEGAN_gen_0092.png" width="256"><img src="fig_BEGAN_gen_0094.png" width="256"><img src="fig_BEGAN_gen_0096.png" width="256">
