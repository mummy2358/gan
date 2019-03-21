import cv2
import numpy as np


def unpickle(filename):
	import pickle
	with open(filename, 'rb') as fo:
		dictfile = pickle.load(fo, encoding='bytes')
	return dictfile
	
def shuffle_together(arr1,arr2):
	c=list(zip(arr1,arr2))
	np.random.shuffle(c)
	arr_1,arr_2=zip(*c)
	return arr_1,arr_2
		
def convert_to_images(imgarr):
	# convert image arrays into RGB images
	imgs=[]
	for arr in imgarr:
		img=np.reshape(arr,[3,32,32])
		img=np.swapaxes(img,0,1)
		img=np.swapaxes(img,1,2)
		imgs.append(img)
	return imgs

class loader_cifar100:
	'''
	cifar100: 50000*3072
	'''
	def __init__(self,filedir="./",train_val_ratio=0.9):
		self.train=unpickle(filedir+"train")
		self.test=unpickle(filedir+"test")
		
		# split train and val from train
		self.train_data=convert_to_images(self.train[b'data'])
		self.train_labels=self.train[b'fine_labels']
		self.train_data,self.train_labels=shuffle_together(self.train_data,self.train_labels)

		samnum=len(self.train_data)
		self.val_data=self.train_data[int(train_val_ratio*samnum):]
		self.val_labels=self.train_labels[int(train_val_ratio*samnum):]
		self.train_data=self.train_data[:int(train_val_ratio*samnum)]
		self.train_labels=self.train_labels[:int(train_val_ratio*samnum)]
		
		self.test_data=convert_to_images(self.test[b'data'])
		self.test_labels=self.test[b'fine_labels']
		self.category_num=100
		
		# These are indices for gathering batch data
		self.indices=list(range(len(self.train_data)))
		self.batch_counter=0
		self.sample_counter=0
		
	def train_next_batch(self,batch_size=100,batch_per_epoch=0):
		batch_x=[]
		batch_y=[]
		np.random.shuffle(self.indices)
		while True:
			img=self.train_data[self.indices[self.sample_counter]]
			label=np.eye(self.category_num)[self.train_labels[self.indices[self.sample_counter]]]
			batch_x.append(img)
			batch_y.append(label)
			
			self.sample_counter+=1
			if self.sample_counter%batch_size==0 or self.sample_counter==len(self.indices):
				yield [batch_x,batch_y]
				batch_x=[]
				batch_y=[]
				self.batch_counter+=1
			if batch_per_epoch>0 and self.batch_counter==batch_per_epoch or self.sample_counter==len(self.indices):
				np.random.shuffle(self.indices)
				self.sample_counter=0
	
	def load_val(self):
		# create one hot labels and return both dataset and labels
		return self.val_data,np.eye(self.category_num)[self.val_labels,:]

	def load_test(self):
		# create one hot labels and return both dataset and labels
		return self.test_data,np.eye(self.category_num)[self.test_labels,:]


class loader_cifar10:
	'''
	cifar10: 50000*3072
	'''
	def __init__(self,filedir="./",train_val_ratio=0.9):
		self.train=[unpickle(filedir+"data_batch_"+str(i+1)) for i in range(5)]
		self.test=unpickle(filedir+"test_batch")
		
		# split train and val from train
		self.train_data=convert_to_images(self.train[0][b'data'])
		self.train_labels=self.train[0][b'labels']
		for i in range(1,5):
			self.train_data=np.concatenate((self.train_data,convert_to_images(self.train[i][b'data'])),axis=0)
			self.train_labels=np.concatenate((self.train_labels,self.train[i][b'labels']),axis=0)
		self.train_data,self.train_labels=shuffle_together(self.train_data,self.train_labels)
		samnum=len(self.train_data)
		self.val_data=self.train_data[int(train_val_ratio*samnum):]
		self.val_labels=self.train_labels[int(train_val_ratio*samnum):]
		self.train_data=self.train_data[:int(train_val_ratio*samnum)]
		self.train_labels=self.train_labels[:int(train_val_ratio*samnum)]
		
		self.test_data=convert_to_images(self.test[b'data'])
		self.test_labels=self.test[b'labels']
		self.category_num=10
		
		# These are indices for gathering batch data
		self.indices=list(range(len(self.train_data)))
		self.batch_counter=0
		self.sample_counter=0
		
	def train_next_batch(self,batch_size=100,batch_per_epoch=0):
		batch_x=[]
		batch_y=[]
		np.random.shuffle(self.indices)
		while True:
			img=self.train_data[self.indices[self.sample_counter]]
			label=np.eye(self.category_num)[self.train_labels[self.indices[self.sample_counter]]]
			batch_x.append(img)
			batch_y.append(label)
			
			self.sample_counter+=1
			if self.sample_counter%batch_size==0 or self.sample_counter==len(self.indices):
				yield [batch_x,batch_y]
				batch_x=[]
				batch_y=[]
				self.batch_counter+=1
			if batch_per_epoch>0 and self.batch_counter==batch_per_epoch or self.sample_counter==len(self.indices):
				np.random.shuffle(self.indices)
				self.sample_counter=0
	
	def load_val(self):
		# create one hot labels and return both dataset and labels
		return self.val_data,np.eye(self.category_num)[self.val_labels,:]

	def load_test(self):
		# create one hot labels and return both dataset and labels
		return self.test_data,np.eye(self.category_num)[self.test_labels,:]



