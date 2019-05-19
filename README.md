# AM-GAN

This is a clean implementation of AM-GAN. It also provides modularized implementation of evaluation metrics including AM-Score, Inception-Score and FID, along with a simple yet powerful logger system. 

It was written with tensorflow-1.5 and python3.5. You can setting up the required environments according to build_env.sh. Note that the network structures and hyper-parameters are slight different from [AM-GAN](https://github.com/ZhimingZhou/AM-GAN).

In traditional GANs (including vanilla GAN, least-square GAN etc.), balance of G and D is very important. In practice, we usually adjust the hyper-parameters such that the discriminator is hard to distinguish the real and fake samples during the training process. 

A theoretical explanation and solution of the above issue can be found in https://arxiv.org/abs/1807.00751.

```
The gradient from traditional GANs can be unwarranted and does not guarantee convergence (though have global minimum at p_g=p_data).

Empirically, when the discriminator is hard to distinguish the real and fake samples, the gradient from traditional GANs is more reliable.

To ensure meaningful gradient direction and convergence for the training of GANs, it requires defining the D(x) not only on p_g and p_data but also the whole space X or at least a path from p_g
to p_data. Lipschitz constraint is one of the tools that can achieve the required properties. GANs that holds this property includes Wasserstein GAN, Coulomb GAN, etc. 

See https://arxiv.org/abs/1807.00751 for more details and generalized discussion on the properties of Lipschitz in GANs.
```
