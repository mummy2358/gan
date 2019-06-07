import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def shuffle_together(arr1,arr2):
	c=list(zip(arr1,arr2))
	np.random.shuffle(c)
	arr_1,arr_2=zip(*c)
	return arr_1,arr_2

def acc(label,pred):
	# input predictions and labels and calculate accuracy
	correct=0
	for i in range(len(label)):
		if np.argmax(label[i])==np.argmax(pred[i]):
			correct+=1
	res=correct/len(label)
	return res

def show_batchimage(features,h,w):
	# display images in subplot form with given rows,cols
	# images in shape of [batch,height,width,3]
	for i in range(h):
		for j in range(w):
			idx=i*w+j
			if idx<np.shape(features)[0]:
				plt.axis('off')
				plt.subplot(h,w,idx+1)
				plt.imshow(features[idx,:,:,:])
	plt.axis('off')
	plt.show()
