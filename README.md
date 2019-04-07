generative adversial network using tf.nn from scratch on cifar10
some of the outputs after 20 epochs:(10000 adversial iteration, 100 batch size)

![test1](https://github.com/mummy2358/gan/blob/master/test1.png)
![test2](https://github.com/mummy2358/gan/blob/master/test1_1.png)
![loss curve1](https://github.com/mummy2358/gan/blob/master/Figure_1.png)
![loss curve2](https://github.com/mummy2358/gan/blob/master/prob_40000.png)

![test image](https://github.com/mummy2358/gan/blob/master/test2_copy.png)

I tried both the fully connected and convolutional version models of the original auther. The fully connected version seems to have less problem of mode collapse( outputs look almost the same no matter what random number you put in ). 1e-5 seems to be a fair learning rate for adversial to happen quickly. 

Also tried out DCGAN, seems easily get into mode collapse. And all of them end up with too strong descriminator ( D(x)->1 and D(G(z))->0 ).
______________________
latest update:
outputs of DCGAN after 100 epochs:
![test1_n](https://github.com/mummy2358/gan/blob/master/test1.png)
![test2_n](https://github.com/mummy2358/gan/blob/master/test2.png)
![test3_n](https://github.com/mummy2358/gan/blob/master/test3.png)
same output problem comes from np.random.rand() which is substituted with np.random.uniform()

