# this is a test program of gan on cifar 100
# mym March 2019
# tensorflow version: 1.12.0
# using only basic tensorflow API
# may later be structured in a better way
import tensorflow as tf
import numpy as np
import data_loader
import cv2
import os

def conv2d_layer(inputs,kernel_size=3,filters=64,strides=[2,2],kernel_init=None,stddev=1e-3,use_bias=True,bias_init=None,activation=None,name="conv0"):
	# kernel_size and filters are both integers
	# inputs is a tensor of [batch,height,width,channels]
	with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
		# auto_reuse also reuse scope
		if not kernel_init:
			kernels=tf.get_variable(initializer=tf.random.truncated_normal(shape=[kernel_size,kernel_size,inputs.get_shape()[-1].value,filters],stddev=stddev),dtype=tf.float32,name="kernels")
		else:
			kernels=tf.get_variable(initializer=kernel_init,dtype=tf.float32,name="kernels")
		if use_bias:
			if not bias_init:
				bias=tf.get_variable(initializer=tf.random.truncated_normal(shape=[filters],stddev=stddev),dtype=tf.float32,name="bias")
			else:
				bias=tf.get_variable(initializer=bias_init,dtype=tf.float32,name="bias")
		res=tf.nn.convolution(inputs,kernels,padding="SAME",strides=strides,name=name)
		res=tf.nn.bias_add(res,bias,data_format="NHWC")
		if activation:
			res=activation(res)
		return res

def fc_layer(inputs,units=100,W_init=None,stddev=1e-3,use_bias=True,bias_init=None,activation=None,name="fc0"):
	# save the first rank and flatten all the rest
	# be sure that inputs have at least rank of 2
	rest=1
	for m in inputs.get_shape()[1:]:
		rest=rest*m.value
	flat=tf.reshape(inputs,shape=[-1,rest])
	with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
		if W_init:
			W=tf.get_variable(initializer=W_init,dtype=tf.float32,name="W")
		else:
			W=tf.get_variable(initializer=tf.random.truncated_normal(shape=[rest,units],stddev=stddev),dtype=tf.float32,name="W")
		if use_bias:
			if bias_init:
				b=tf.get_variable(initializer=bias_init,dtype=tf.float32,name="b")
			else:
				b=tf.get_variable(initializer=tf.random.truncated_normal(shape=[units],stddev=stddev),dtype=tf.float32,name="b")
		
		res=tf.matmul(flat,W)
		res=tf.nn.bias_add(res,b)
		if activation:
			res=activation(res)
		return res

def acc(label,pred):
	# input predictions and labels and calculate accuracy
	correct=0
	for i in range(len(label)):
		if np.argmax(label[i])==np.argmax(pred[i]):
			correct+=1
	res=correct/len(label)
	return res

class generator:
	def __init__(self,name="G"):
		# feed forward neural network used for generating image samples
		# here suppose we just input one random number
		self.name=name
	
	def forward(self,inputs):
		# directly define struct by hand for experiments
		# later should be generalized to random structure
		with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE):
			fc1=fc_layer(inputs,units=10,name="fc1")
			fc2=fc_layer(fc1,units=32*32*10,name="fc2")
			fc2_re=tf.reshape(fc2,shape=[-1,32,32,10],name="reshape")
			conv1=conv2d_layer(fc2_re,kernel_size=3,filters=64,strides=[1,1],activation=tf.nn.relu,name="conv1")
			conv2=conv2d_layer(conv1,kernel_size=3,filters=64,strides=[1,1],activation=tf.nn.relu,name="conv2")
			conv3=conv2d_layer(conv2,kernel_size=3,filters=64,strides=[1,1],activation=tf.nn.relu,name="conv3")
			conv4=conv2d_layer(conv3,kernel_size=3,filters=3,strides=[1,1],activation=tf.nn.relu,name="conv4")
			return conv4


class discriminator:
	def __init__(self,name="D"):
		# feed forward nn for judging if a sample comes from generator or real samples
		self.name=name
	
	def forward(self,inputs):
		# same as generator; should later be generalized
		with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE):
			conv1=conv2d_layer(inputs,kernel_size=3,filters=64,strides=[1,1],activation=tf.nn.relu,name="conv1")
			conv2=conv2d_layer(conv1,kernel_size=3,filters=64,strides=[1,1],activation=tf.nn.relu,name="conv2")
			conv3=conv2d_layer(conv2,kernel_size=3,filters=64,strides=[1,1],activation=tf.nn.relu,name="conv3")
			fc1=fc_layer(conv3,units=10,activation=tf.nn.sigmoid,name="fc1")
			fc2=fc_layer(fc1,units=1,activation=tf.nn.sigmoid,name="fc2")
			return fc2

class gan:
	def __init__(self,name="GAN_cifar10"):
		self.name=name
		with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE):
			self.x=tf.placeholder(shape=[None,32,32,3],dtype=tf.float32)
			self.z=tf.placeholder(shape=[None,10],dtype=tf.float32)
			self.D=discriminator(name="D")
			self.G=generator(name="G")
			self.build()
		
	def build(self):
		# three losses to use
		# discriminator loss needs maximizing
		# generator loss needs minimizing
		self.D_loss=tf.reduce_mean(tf.log(self.D.forward(self.x))+tf.log(1-self.D.forward(self.G.forward(self.z))))
		self.G_loss_pre=tf.reduce_mean(self.D.forward(self.G.forward(self.z)))
		self.G_loss_post=tf.reduce_mean(tf.log(1-self.D.forward(self.G.forward(self.z))))
		self.test_term1=tf.reduce_mean(tf.log(self.D.forward(self.G.forward(self.z))))
		self.test_term2=tf.reduce_mean(tf.log(self.D.forward(self.x)))
		self.img=self.G.forward(self.z)
		self.pred=self.D.forward(self.x)

gan0=gan(name="gan0")


D_train_step=tf.train.AdamOptimizer(learning_rate=1e-4).minimize(-gan0.D_loss,var_list=tf.trainable_variables(scope=gan0.name+"/D"))
G_train_step_pre=tf.train.AdamOptimizer(learning_rate=1e-4).minimize(-gan0.G_loss_pre,var_list=tf.trainable_variables(scope=gan0.name+"/G"))
G_train_step_post=tf.train.AdamOptimizer(learning_rate=1e-4).minimize(gan0.G_loss_post,var_list=tf.trainable_variables(scope=gan0.name+"/G"))

init=tf.global_variables_initializer()

## data loader
loader0=data_loader.loader_cifar10(filedir="./cifar-10-batches-py/")
loader0_iterator=loader0.train_next_batch(batch_size=100)
val_dataset=loader0.load_val()
test_dataset=loader0.load_test()


## saving and restoring
savedir="./models"
saver=tf.train.Saver()
# default epoch should be set to 0; otherwise start from the given
restore_epoch=0
save_iter=200


## training and validation
train_num=45000
batch_size=100
batch_per_epoch=train_num//batch_size
max_epoch=10000
sess=tf.Session()
sess.run(init)

div_epoch=5000
k=1

test_epoch=100
test_num=3

random_ratio=1

print("trainable variables:")
print(tf.trainable_variables(scope=gan0.name+"/D"))
print(tf.trainable_variables(scope=gan0.name+"/G"))


if restore_epoch:
	print("restoring model...")
	saver.restore(sess,savedir+"/epoch_"+str(restore_epoch)+".ckpt")

## start training
for e in range(restore_epoch,max_epoch):
	print("*************")
	print("epoch "+str(e+1)+":")
	for i in range(k):
		# update discriminator
		# k is the times of updating discriminator
		batch=next(loader0_iterator)
		batch_x=batch[0]
		batch_z=np.random.rand(*[len(batch_x),*gan0.z.get_shape()[1:]])
		batch_z=batch_z*random_ratio
		
		_,batch_D_loss=sess.run([D_train_step,gan0.D_loss],feed_dict={gan0.x:batch_x,gan0.z:batch_z})
		print("batch_D_loss: "+str(batch_D_loss))
	
	# update generator G
	batch_z=np.random.rand(*[batch_size,*gan0.z.get_shape()[1:]])
	batch_z=batch_z*random_ratio
	print(np.shape(batch_z))
	if e<div_epoch:
		# use G_loss_pre here
		_,batch_G_loss_pre=sess.run([G_train_step_pre,gan0.G_loss_pre],feed_dict={gan0.z:batch_z})
		print("batch_G_loss_pre: "+str(batch_G_loss_pre))
		term1,term2=sess.run([gan0.test_term1,gan0.test_term2],feed_dict={gan0.z:batch_z,gan0.x:batch_x})
		print("term1: "+str(term1))
		print("term2: "+str(term2))
	else:
		# use G_loss_post here
		_,batch_G_loss_post=sess.run([G_train_step_post,gan0.G_loss_post],feed_dict={gan0.z:batch_z})
		print("batch_G_loss_post: "+str(batch_G_loss_post))
	#writer = tf.summary.FileWriter("./output", sess.graph)	
	if not (e+1)%save_iter:
		print("saving model...")
		saver.save(sess,savedir+"/epoch_"+str(e+1)+".ckpt")
	if not (e+1)%test_epoch:
	
		r=np.random.rand(test_num,*gan0.z.get_shape()[1:])
		img=sess.run(gan0.img,feed_dict={gan0.z:r})
		for i in range(test_num):
			cv2.imshow("win0",img[i])
			cv2.waitKey()
			cv2.destroyAllWindows()

