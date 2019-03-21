# this is a test program of gan on cifar 10
# mym March 2019
# tensorflow version: 1.12.0
# using only basic tensorflow API
import tensorflow as tf
import numpy as np
import data_loader


def conv2d_layer(inputs,kernel_size=3,filters=64,strides=[2,2],kernel_init=None,stddev=1e-3,use_bias=True,bias_init=None,activation=None,name="conv0"):
	# kernel_size and filters are both integers
	# inputs is a tensor of [batch,height,width,channels]
	with tf.name_scope(name) as scope:
		if not kernel_init:
			kernels=tf.Variable(tf.random.truncated_normal(shape=[kernel_size,kernel_size,inputs.get_shape()[-1].value,filters],stddev=stddev),dtype=tf.float32,name="kernels")
		else:
			kernels=tf.Variable(kernel_init,dtype=tf.float32,name="kernels")
		if use_bias:
			if not bias_init:
				bias=tf.Variable(tf.random.truncated_normal(shape=[filters],stddev=stddev),dtype=tf.float32,name="bias")
			else:
				bias=tf.Variable(bias_init,dtype=tf.float32,name="bias")
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
	flat=tf.reshape(inputs,shape=[-1,m])
	with tf.name_scope(name) as scope:
		if W_init:
			W=tf.Variable(W_init,dtype=tf.float32,name="W")
		else:
			W=tf.Variable(tf.random.truncated_normal(shape=[rest,units],stddev=stddev),dtype=tf.float32,name="W")
		if use_bias:
			if bias_init:
				b=tf.Variable(bias_init,dtype=tf.float32,name="b")
			else:
				b=tf.Variable(tf.random.truncated_normal(shape=[units],stddev=stddev),dtype=tf.float32,name="b")
		
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


x=tf.placeholder(shape=[None,32,32,3],dtype=tf.float32)
y=tf.placeholder(shape=[None,10],dtype=tf.float32)

conv1=conv2d_layer(x,kernel_size=7,activation=tf.nn.relu)
pool1=tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
conv2=conv2d_layer(pool1,kernel_size=5,activation=tf.nn.relu)
pool2=tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
conv3=conv2d_layer(pool2,kernel_size=3,filters=64,activation=tf.nn.relu)
pool3=tf.nn.max_pool(conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
conv4=conv2d_layer(pool3,kernel_size=3,filters=64,activation=tf.nn.relu)
pool4=tf.nn.max_pool(conv4,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

fc1=fc_layer(pool4,units=1024,activation=tf.nn.relu)
fc2=fc_layer(fc1,units=10)

loss=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=fc2))


train_step=tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss)
init=tf.global_variables_initializer()


#data loader
loader0=data_loader.loader_cifar10(filedir='./cifar-10-batches-py/')
loader0_iterator=loader0.train_next_batch(batch_size=100)
val_dataset=loader0.load_val()
test_dataset=loader0.load_test()
# training and validation
train_num=45000
batch_size=100
batch_per_epoch=train_num//batch_size
max_epoch=1000
val_iter=10
test_iter=100
sess=tf.Session()
sess.run(init)

for e in range(max_epoch):
	loss_epoch=0
	for b in range(batch_per_epoch):
		batch=next(loader0_iterator)
		_,loss_batch=sess.run([train_step,loss],feed_dict={x:batch[0],y:batch[1]})
		loss_epoch=loss_epoch+loss_batch
	print("epoch "+str(e+1))
	print("average loss: "+str(loss_epoch/train_num))
	if not (e+1)%val_iter:
		val_pred=sess.run(fc2,feed_dict={x:val_dataset[0],y:val_dataset[1]})
		val_acc=acc(label=val_dataset[1],pred=val_pred)
		print("validation acc: "+str(val_acc))
	if not (e+1)%test_iter:
		test_pred=sess.run(fc2,feed_dict={x:test_dataset[0],y:test_dataset[1]})
		test_acc=acc(label=test_pred[1],pred=test_pred)
		print("test acc: "+str(test_acc))


