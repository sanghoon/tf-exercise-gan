# tf-exercise-gan

Tensorflow implementation and benchmark of diffrent GANs


## GAN implementations

- [x] **DCGAN** from 'Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks' (https://arxiv.org/abs/1511.06434)
- [x] **WGAN** from 'Wasserstein GAN' (https://arxiv.org/abs/1701.07875)
- [x] **BEGAN** from 'BEGAN: Boundary Equilibrium Generative Adversarial Networks' (https://arxiv.org/abs/1703.10717)
- [x] **MAD-GAN** from 'Multi-Agent Diverse Generative Adversarial Networks' (https://arxiv.org/abs/1704.02906)
- [x] **GoGAN** from 'Gang of GANs: Generative Adversarial Networks with Maximum Margin Ranking' (https://arxiv.org/abs/1704.04865)
- [ ] ...

## Tasks

- [x] Impl. DCGAN, GoGAN, WGAN
- [x] Impl. BEGAN, MAD-GAN
- [x] Reproduce GANs on MNIST and CelebA datasets
- [x] Impl. training & evaluation on synthetic datasets
- [x] Add sample results
- [ ] Impl. inference-only codes for GANs (may require refactoring)
- [ ] Impl. better evaluation function for real images (e.g. IvOM, energy dist., ...)
- [ ] Impl. a result logger
- [x] Compare GANs (synthetic)
- [ ] Compare GANs (MNIST and CelebA dataset)
- [ ] Add more GAN implementations


## Experiments & Benchmarks

170718 / [Comparison of different GAN models on synthetic datasets](assets/170718_synthetic/report_synthetic.md)

- Done without any hyper-parameter search.
- MAD-GAN worked best in the tested datasets.
- ![MADGAN_Spiral](assets/170718_synthetic/MADGAN_SynSpiral_toydisc_toydisc_LR=0.0001_NGEN=8.gif)

170718 / [Sample results on MNIST dataset](assets/170718_mnist/samples.md)

- ![WGAN_MNIST](assets/170718_mnist/WGAN_mnist_099.png)

170809 / [Sample results on CelebA dataset](assets/170809_celeba/samples.md)
- ![BEGAN_CELEBA](assets/170809_celeba/fig_BEGAN_gen_0096.png)
