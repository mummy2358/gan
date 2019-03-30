generative adversial network using tf.nn from scratch on cifar10
some of the outputs after 20 epochs:(10000 adversial iteration, 100 batch size)

![test1](https://github.com/mummy2358/gan/blob/master/test1.png)
![test2](https://github.com/mummy2358/gan/blob/master/test1_1.png)
![loss curve](https://github.com/mummy2358/gan/blob/master/Figure_1.png)

I tried both the fully connected and convolutional version models of the original auther. The fully connected version seems to have less problem of mode collapse( outputs look almost the same no matter what random number you put in ). 1e-5 seems to be a fair learning rate for adversial to happen quickly. 

Also tried out DCGAN, seems easily get into mode collapse. And all of them end up with too strong descriminator ( D(x)->1 and D(G(z))->0 ).
