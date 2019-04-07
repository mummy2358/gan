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

def deconv2d_layer(inputs,kernel_size=3,filters=64,strides=[1,2,2,1],kernel_init=None,stddev=1e-3,use_bias=True,bias_init=None,activation=None,name="deconv0"):
	# kernel_size and filters are both integers
	# inputs is a tensor of [batch,height,width,channels]
	inputs_shape=inputs.get_shape()
	with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
		# auto_reuse also reuse scope
		if not kernel_init:
			kernels=tf.get_variable(initializer=tf.random.truncated_normal(shape=[kernel_size,kernel_size,filters,inputs.get_shape()[-1].value],stddev=stddev),dtype=tf.float32,name="kernels")
		else:
			kernels=tf.get_variable(initializer=kernel_init,dtype=tf.float32,name="kernels")
		if use_bias:
			if not bias_init:
				bias=tf.get_variable(initializer=tf.random.truncated_normal(shape=[filters],stddev=stddev),dtype=tf.float32,name="bias")
			else:
				bias=tf.get_variable(initializer=bias_init,dtype=tf.float32,name="bias")
		res=tf.nn.conv2d_transpose(inputs,kernels,output_shape=tf.stack([tf.shape(inputs)[0]*strides[0],inputs_shape[1]*strides[1],inputs_shape[2]*strides[2],filters*strides[3]]),padding="SAME",strides=strides,name=name)
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


def minibatch_layer(inputs,B,C,stddev,name="minibatch_layer0"):
	# inputs is a [batch,v] shape tensor
	sh=inputs.get_shape()
	n=sh[0]
	v=sh[1]
	with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
		T=tf.get_variable(initializer=tf.random.truncated_normal(shape=[v.value,B,C],stddev=stddev),dtype=tf.float32,name="T")
		M=tf.tensordot(inputs,T,axes=[-1,0])
		o=[]
		for i in range(n):
			o.append([])
			for b in range(B):
				o[i].append(tf.reduce_sum(tf.math.exp(-tf.pow(M[i,b]-M[:,b],2))))
			o[i]=tf.stack(o[i])
		o=tf.stack(o)
		return o

def acc(label,pred):
	# input predictions and labels and calculate accuracy
	correct=0
	for i in range(len(label)):
		if np.argmax(label[i])==np.argmax(pred[i]):
			correct+=1
	res=correct/len(label)
	return res

def truncate_imgs(imgs):
	# convert image matrices to uint8 and truncate to 0-255
	tr=np.array(imgs)
	tr=np.clip(imgs,a_min=0,a_max=255)
	tr=np.uint8(tr)
	tr.astype(np.uint8)
	return tr

def normalize(inputs,maxv=255,minv=0):
	# transformed to -1 and 1
	inputs=np.array(inputs)
	if maxv>minv:
		res=(inputs-minv)/(maxv-minv)*2-1
		return res
	else:
		print("invalid given maxv and minv")
def normalize01(inputs,maxv=255,minv=0):
	# transformed to 0 and 1
	inputs=np.array(inputs)
	if maxv>minv:
		res=(inputs-minv)/(maxv-minv)
		return res
	else:
		print("invalid given maxv and minv")


def add_noise(inputs,noise_range=0.5):
	# add uniform distributed noise to input images on -1 and 1
	res=inputs+np.random.rand(*np.shape(inputs))*noise_range*2-1
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
			fc1=fc_layer(inputs,units=256*2*2,activation=None,name="fc1")
			fc1_re=tf.reshape(fc1,shape=[-1,2,2,256],name="reshape")
			deconv1=deconv2d_layer(fc1_re,kernel_size=5,filters=128,strides=[1,2,2,1],activation=tf.nn.leaky_relu,name="deconv1")
			deconv2=deconv2d_layer(deconv1,kernel_size=5,filters=64,strides=[1,2,2,1],activation=tf.nn.leaky_relu,name="deconv2")
			deconv3=deconv2d_layer(deconv2,kernel_size=5,filters=32,strides=[1,2,2,1],activation=tf.nn.leaky_relu,name="deconv3")
			deconv4=deconv2d_layer(deconv3,kernel_size=5,filters=3,strides=[1,2,2,1],activation=tf.nn.tanh,name="deconv4")
			return deconv4


class discriminator:
	def __init__(self,name="D"):
		# feed forward nn for judging if a sample comes from generator or real samples
		self.name=name
	
	def forward(self,inputs):
		# same as generator; should later be generalized
		with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE):
			inputs_drop=tf.nn.dropout(inputs,keep_prob=1.0)
			
			conv1=conv2d_layer(inputs_drop,kernel_size=5,filters=32,strides=[2,2],activation=tf.nn.leaky_relu,name="conv1")
			#conv1=tf.nn.max_pool(conv1,ksize=[1,4,4,1],strides=[1,2,2,1],padding="SAME",name="pool1")
			#conv1=tf.contrib.layers.maxout(conv1,num_units=int(32/2))
			
			conv2=conv2d_layer(conv1,kernel_size=5,filters=64,strides=[2,2],activation=tf.nn.leaky_relu,name="conv2")
			#conv2=tf.nn.max_pool(conv2,ksize=[1,4,4,1],strides=[1,2,2,1],padding="SAME",name="pool2")
			#conv2=tf.contrib.layers.maxout(conv2,num_units=int(32/2))
			
			conv3=conv2d_layer(conv2,kernel_size=5,filters=128,strides=[2,2],activation=tf.nn.leaky_relu,name="conv3")
			#conv3=tf.nn.max_pool(conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME",name="pool3")
			#conv3=tf.contrib.layers.maxout(conv3,num_units=int(64/2))
			
			conv4=conv2d_layer(conv3,kernel_size=5,filters=256,strides=[2,2],activation=tf.nn.leaky_relu,name="conv4")
			
			conv5=conv2d_layer(conv4,kernel_size=5,filters=512,strides=[2,2],activation=tf.nn.leaky_relu,name="conv5")
			
			#fc1=fc_layer(conv3,units=100,activation=tf.nn.leaky_relu,name="fc1")
			#fc1=tf.contrib.layers.maxout(fc1,num_units=int(500/5))
			
			fc1=fc_layer(conv5,units=1,activation=tf.nn.sigmoid,name="fc1")
			return fc1

class gan:
	def __init__(self,name="GAN_cifar10"):
		self.name=name
		with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE):
			self.x=tf.placeholder(shape=[None,32,32,3],dtype=tf.float32)
			self.z=tf.placeholder(shape=[None,100],dtype=tf.float32)
			self.D=discriminator(name="D")
			self.G=generator(name="G")
			self.build()
		
	def build(self):
		# three losses to use
		# discriminator loss needs maximizing
		# generator loss needs minimizing
		self.D_loss=tf.reduce_mean(tf.log(self.D.forward(self.x)+1e-30)+tf.log(1-self.D.forward(self.G.forward(self.z))+1e-30))
		self.G_loss_pre=tf.reduce_mean(tf.log(self.D.forward(self.G.forward(self.z))+1e-30))
		self.G_loss_post=tf.reduce_mean(tf.log(1-self.D.forward(self.G.forward(self.z))+1e-30))
		
		self.x_loss=tf.reduce_mean(tf.log(self.D.forward(self.x)+1e-30))
		self.test_term1=tf.reduce_mean(self.D.forward(self.x))		
		self.test_term2=tf.reduce_mean(self.D.forward(self.G.forward(self.z)))
		self.img=self.G.forward(self.z)
		self.pred=self.D.forward(self.x)

gan0=gan(name="gan0")


D_train_step=tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.5,beta2=0.999).minimize(-gan0.D_loss,var_list=tf.trainable_variables(scope=gan0.name+"/D"))
G_train_step_pre=tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.5,beta2=0.999).minimize(-gan0.G_loss_pre,var_list=tf.trainable_variables(scope=gan0.name+"/G"))
G_train_step_post=tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.5,beta2=0.999).minimize(gan0.G_loss_post,var_list=tf.trainable_variables(scope=gan0.name+"/G"))

init=tf.global_variables_initializer()

## data loader
batch_size=100
loader0=data_loader.loader_cifar10(filedir="./cifar-10-batches-py/")
loader0_iterator=loader0.train_next_batch(batch_size=batch_size)
val_dataset=loader0.load_val()
test_dataset=loader0.load_test()


## saving and restoring
savedir="./models"
saver=tf.train.Saver()
# default epoch should be set to 0; otherwise start from the given
restore_epoch=37500
save_iter=500


## training and validation
train_num=45000
batch_per_epoch=train_num//batch_size
max_epoch=100000
sess=tf.Session()
sess.run(init)

div_epoch=0
d=1
g=1

test_epoch=10000
test_num=3

random_ratio=1

term1_list=[]
term2_list=[]
D_loss_list=[]
G_loss_list=[]

print("trainable variables:")
print(tf.trainable_variables(scope=gan0.name+"/D"))
print(tf.trainable_variables(scope=gan0.name+"/G"))


if restore_epoch>1:
	print("restoring model...")
	saver.restore(sess,savedir+"/epoch_"+str(restore_epoch)+".ckpt")

## start training
for e in range(restore_epoch,max_epoch):
	print("*************")
	print("epoch "+str(e+1)+":")
	if not (e+1)%test_epoch:
		#np.random.seed(100)
		r=np.random.uniform(low=-1,high=1,size=[test_num,*gan0.z.get_shape()[1:]])
		img=sess.run(gan0.img,feed_dict={gan0.z:r})
		tr=(img+1)/2*255
		tr=truncate_imgs(tr)
		for i in range(test_num):
			cv2.imshow("win0",tr[i])
			cv2.imwrite("./test"+str(i+1)+".png",tr[i])
			cv2.waitKey()
			cv2.destroyAllWindows()
	for i in range(d):
		# update discriminator
		# k is the times of updating discriminator
		batch=next(loader0_iterator)
		batch_x=batch[0]

		# preprocess inputs
		batch_x=normalize(batch_x)
		
		#batch_x=add_noise(batch_x,0.1)
		#batch_x=np.clip(batch_x,a_min=-1,a_max=1)
		
		batch_z=np.random.uniform(low=-1,high=1,size=[len(batch_x),*gan0.z.get_shape()[1:]])
		batch_z=batch_z*random_ratio
		
		# update D
		_,batch_D_loss=sess.run([D_train_step,gan0.D_loss],feed_dict={gan0.x:batch_x,gan0.z:batch_z})
	for j in range(g):
		# update G
		batch_z=np.random.uniform(low=-1, high=1,size=[batch_size,*gan0.z.get_shape()[1:]])
		batch_z=batch_z*random_ratio
		if e<div_epoch:
			# use G_loss_pre here
			_,batch_G_loss_pre=sess.run([G_train_step_pre,gan0.G_loss_pre],feed_dict={gan0.z:batch_z})
		else:
			# use G_loss_post here
			_,batch_G_loss_post=sess.run([G_train_step_post,gan0.G_loss_post],feed_dict={gan0.z:batch_z})
	print("batch_D_loss: "+str(batch_D_loss))
	
	if e<div_epoch:
		print("batch_G_loss_pre: "+str(batch_G_loss_pre))
	else:
		print("batch_G_loss_post: "+str(batch_G_loss_post))
	# calculate loss after one complete epoch
	
	# the terms are calculated after a complete adversial epoch
	term1,term2,x_loss,D_loss_t,G_loss_t=sess.run([gan0.test_term1,gan0.test_term2,gan0.x_loss,gan0.D_loss,gan0.G_loss_pre],feed_dict={gan0.z:batch_z,gan0.x:batch_x})
	print("x loss: "+str(x_loss))
	term1_list.append(term1)
	term2_list.append(term2)
	D_loss_list.append(D_loss_t)
	G_loss_list.append(G_loss_t)
	print("term1: "+str(term1))
	print("term2: "+str(term2))
	# writer = tf.summary.FileWriter("./output", sess.graph)	
	if not (e+1)%save_iter:
		print("saving model...")
		saver.save(sess,savedir+"/epoch_"+str(e+1)+".ckpt")
		np.save("./term1.npy",np.array(term1_list))
		np.save("./term2.npy",np.array(term2_list))
		np.save("./D_loss.npy",np.array(D_loss_list))
		np.save("./G_loss.npy",np.array(G_loss_list))
