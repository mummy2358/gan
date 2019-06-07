import tensorflow as tf
import numpy as np
import utils

generating_num=100

sess=tf.Session()

saver=tf.train.import_meta_graph("./models/epoch_69500.ckpt.meta")
saver.restore(sess,"./models/epoch_69500.ckpt")


graph=tf.get_default_graph()

namedict=np.load("./namedict.npy").item()

outputs=graph.get_tensor_by_name(namedict["outputs"])
inputs=graph.get_tensor_by_name(namedict["inputs"])

z=np.random.uniform(low=-1,high=1,size=[generating_num,*inputs.get_shape()[1:]])
generated=sess.run(outputs,feed_dict={inputs:z})

utils.show_batchimage(generated,10,10)
