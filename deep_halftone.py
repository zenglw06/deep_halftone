import tensorflow as tf
from PIL import Image
import numpy as np 
import os
import pdb
from tensorflow.contrib import layers
from tensorflow.contrib.framework import arg_scope
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
COVER_TRAIN = '/home/titan/zenglw/cover/HT_cover/HT_cover/'
STEGO_TRAIN =  '/home/titan/zenglw/w3r0.125/w3r0.125/'

COVER_VALID = '/home/titan/zenglw/cover/HT_cover/valid/'
STEGO_VALID =  '/home/titan/zenglw/w3r0.125/valid/'
	
COVER_TEST = '/home/titan/zenglw/cover/HT_cover/test/'
STEGO_TEST =  '/home/titan/zenglw/w3r0.125/test/'
BATCH_SIZE = 16
MAX_STEP = 10000

TRAIN_INTERVALS = 100
SAVE_INTERVALS = 100
VALID_INTERVALS =100

boundaries = [7000]     # learning rate adjustment at iteration 400K
values = [0.001, 0.0001]  # learning rates
PICW = 256
PICH = 256
cnt = 0
def read_data(cover_path, stego_path):
	labels = []
	pics = []
	cnt = 0
	for fname in os.listdir(cover_path):
		cnt = cnt+1
		print('%d-th pic' % cnt)
		fpath = os.path.join(cover_path,fname)
		image = Image.open(fpath)
		data = np.array(image,dtype=np.float32)
		data = data.reshape((PICW,PICH,1))
		data = tf.convert_to_tensor(data,dtype = tf.float32)
		label =int(0)
		label = tf.convert_to_tensor(label,dtype = tf.int64)
		labels.append(label)
		pics.append(data)
		#if cnt>=100:
		#	break
	cnt = 0
	#covers = list(covers)
	for fname in os.listdir(stego_path):
		cnt = cnt+1
		print('%d-th stego pic' % cnt)
		fpath = os.path.join(stego_path,fname)
		image = Image.open(fpath)
		data = np.array(image, dtype=np.float32)
		data = data.reshape((PICW,PICH,1))
		data = tf.convert_to_tensor(data,dtype = tf.float32)
		pics.append(data)
		label = int(1)
		label = tf.convert_to_tensor(label,dtype = tf.int64)
		labels.append(label)
		#if cnt>=100:
		#	break
	pics = list(pics)
	return pics,labels,cnt
def get_batch_data(batch_size,cover_path,stego_path):
	pics,labels,cnt = read_data(cover_path,stego_path)
	#pdb.set_trace()
	input_queue = tf.train.slice_input_producer([pics, labels],shuffle = True)
	image_batch,label_batch = tf.train.batch(input_queue,batch_size=batch_size)
	return image_batch,label_batch,cnt



x = tf.placeholder(tf.float32,[None,256,256,1])
x = tf.cast(tf.transpose(x,[0,3,1,2]),tf.float32)
y = tf.placeholder(tf.int64,[None])
reduction_axis = [2,3]

train_image_batch, train_label_batch,cnt=get_batch_data(batch_size=BATCH_SIZE,cover_path=COVER_TRAIN,stego_path=STEGO_TRAIN)
assert cnt!=0, "train set has no pics"
print("=======load train of %d pics success=======" %cnt)
train_image_batch  = tf.cast(tf.transpose(train_image_batch,[0,3,1,2]),tf.float32)

valid_image_batch, valid_label_batch,cnt = get_batch_data(batch_size=BATCH_SIZE,cover_path=COVER_VALID,stego_path=STEGO_VALID)
assert cnt!=0, "train set has no pics"
print("=======load valid of %d pics success=======" %cnt)
valid_image_batch  = tf.cast(tf.transpose(valid_image_batch,[0,3,1,2]),tf.float32)

test_image_batch, test_label_batch,cnt = get_batch_data(batch_size=BATCH_SIZE,cover_path=COVER_TEST,stego_path=STEGO_TEST)
assert cnt!=0, "train set has no pics"
print("=======load test of %d pics success=======" %cnt)
test_image_batch  = tf.cast(tf.transpose(test_image_batch,[0,3,1,2]),tf.float32)

test_step = (cnt / BATCH_SIZE)


with arg_scope([layers.conv2d],num_outputs=16,kernel_size=3, stride = 1, padding = 'SAME',
				activation_fn= None,weights_initializer=layers.variance_scaling_initializer(), data_format ='NCHW',
				weights_regularizer=layers.l2_regularizer(2e-4), biases_initializer=tf.constant_initializer(0.2),biases_regularizer=None),\
	arg_scope([layers.batch_norm],decay=0.9,center=True,scale=True,is_training=True,updates_collections=None,fused=True,data_format ='NCHW'),\
	arg_scope([layers.avg_pool2d],kernel_size=[3,3],stride=[2,2],padding='SAME',data_format='NCHW' ):
	with tf.variable_scope('Layer1'):
		conv = layers.conv2d(x, num_outputs=64, kernel_size=3)
		actv = tf.nn.relu(layers.batch_norm(conv))
	with tf.variable_scope('Layer2'):
		conv = layers.conv2d(actv)
		actv = tf.nn.relu(layers.batch_norm(conv))
	with tf.variable_scope('Layer3'): 
		conv1=layers.conv2d(actv)
		actv1=tf.nn.relu(layers.batch_norm(conv1))
		conv2=layers.conv2d(actv1)
		bn2=layers.batch_norm(conv2)
		res= tf.add(actv, bn2)
	with tf.variable_scope('Layer4'): 
		conv1=layers.conv2d(res)
		actv1=tf.nn.relu(layers.batch_norm(conv1))
		conv2=layers.conv2d(actv1)
		bn2=layers.batch_norm(conv2)
		res= tf.add(res, bn2)
	with tf.variable_scope('Layer5'): 
		conv1=layers.conv2d(res)
		actv1=tf.nn.relu(layers.batch_norm(conv1))
		conv2=layers.conv2d(actv1)
		bn=layers.batch_norm(conv2)
		res= tf.add(res, bn)
	with tf.variable_scope('Layer6'): 
		conv1=layers.conv2d(res)
		actv1=tf.nn.relu(layers.batch_norm(conv1))
		conv2=layers.conv2d(actv1)
		bn=layers.batch_norm(conv2)
		res= tf.add(res, bn)
	with tf.variable_scope('Layer7'): 
		conv1=layers.conv2d(res)
		actv1=tf.nn.relu(layers.batch_norm(conv1))
		conv2=layers.conv2d(actv1)
		bn=layers.batch_norm(conv2)
		res= tf.add(res, bn)
	with tf.variable_scope('Layer8'): 
		convs = layers.conv2d(res, kernel_size=1, stride=2)
		convs = layers.batch_norm(convs)
		conv1=layers.conv2d(res)
		actv1=tf.nn.relu(layers.batch_norm(conv1))
		conv2=layers.conv2d(actv1)
		bn=layers.batch_norm(conv2)
		pool = layers.avg_pool2d(bn)
		res= tf.add(convs, pool)
	with tf.variable_scope('Layer9'):  
		convs = layers.conv2d(res, num_outputs=64, kernel_size=1, stride=2)
		convs = layers.batch_norm(convs)
		conv1=layers.conv2d(res, num_outputs=64)
		actv1=tf.nn.relu(layers.batch_norm(conv1))
		conv2=layers.conv2d(actv1, num_outputs=64)
		bn=layers.batch_norm(conv2)
		pool = layers.avg_pool2d(bn)
		res= tf.add(convs, pool)
	with tf.variable_scope('Layer10'): 
		convs = layers.conv2d(res, num_outputs=128, kernel_size=1, stride=2)
		convs = layers.batch_norm(convs)
		conv1=layers.conv2d(res, num_outputs=128)
		actv1=tf.nn.relu(layers.batch_norm(conv1))
		conv2=layers.conv2d(actv1, num_outputs=128)
		bn=layers.batch_norm(conv2)
		pool = layers.avg_pool2d(bn)
		res= tf.add(convs, pool)
	with tf.variable_scope('Layer11'): 
		convs = layers.conv2d(res, num_outputs=256, kernel_size=1, stride=2)
		convs = layers.batch_norm(convs)
		conv1=layers.conv2d(res, num_outputs=256)
		actv1=tf.nn.relu(layers.batch_norm(conv1))
		conv2=layers.conv2d(actv1, num_outputs=256)
		bn=layers.batch_norm(conv2)
		pool = layers.avg_pool2d(bn)
		res= tf.add(convs, pool)
	with tf.variable_scope('Layer12'): 
		conv1=layers.conv2d(res, num_outputs=512)
		actv1=tf.nn.relu(layers.batch_norm(conv1))
		conv2=layers.conv2d(actv1, num_outputs=512)
		bn=layers.batch_norm(conv2)
		avgp = tf.reduce_mean(bn, reduction_axis,  keep_dims=True )
ip = layers.fully_connected(layers.flatten(avgp), num_outputs=2,
                    activation_fn=None, normalizer_fn=None,
                    weights_initializer=tf.random_normal_initializer(mean=0., stddev=0.01), 
                    biases_initializer=tf.constant_initializer(0.), scope='ip')
oh = tf.one_hot(y,2)
xen_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = oh,logits= ip))
reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.add_n([xen_loss]+reg_loss)
global_step = tf.get_variable('global_step', dtype=tf.int32, shape=[], \
                                  initializer=tf.constant_initializer(0), \
                                  trainable=False)
am = tf.argmax(ip,1)
equal = tf.equal(am,y)
accuracy = tf.reduce_mean(tf.cast(equal,tf.float32))
learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
minimize_op = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step)
train_op = tf.group(minimize_op)

init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
saver =  tf.train.Saver()
train_loss_s=tf.summary.scalar('train_loss',loss)
train_acc_s=tf.summary.scalar('train_accuracy',accuracy)

valid_loss_s = tf.summary.scalar('valid_loss',loss)
valid_acc_s = tf.summary.scalar('valid_acc',accuracy)







with tf.Session() as sess:
	sess.run(init_op)
	coord = tf.train.Coordinator()
	summary_writer = tf.summary.FileWriter('./tmp',sess.graph)
	threads = tf.train.start_queue_runners(sess=sess,coord = coord)
	
	for i in range(1,MAX_STEP):
		print('%d-th step in training' % i)
		val,_y = sess.run([train_image_batch,train_label_batch])
		sess.run(minimize_op,feed_dict={x:val,y:_y})
		anx = sess.run([accuracy,loss],feed_dict={x:val,y:_y})
		print('acc: %.4f   loss: %.4f'%(anx[0],anx[1]))
		if i%TRAIN_INTERVALS==0:
			summary_loss = sess.run(train_loss_s,feed_dict={x:val,y:_y})
			summary_acc = sess.run(train_acc_s,feed_dict={x:val,y:_y})
			summary_writer.add_summary(summary_loss,i)
			summary_writer.add_summary(summary_acc,i)

		if i%VALID_INTERVALS==0:
			valid,_y = sess.run([valid_image_batch,valid_label_batch])
			valid_anx = sess.run([accuracy,loss],feed_dict={x:valid,y:_y})
			summary_valid_loss = sess.run(valid_loss_s,feed_dict={x:valid,y:_y})
			summary_valid_acc = sess.run(valid_acc_s,feed_dict={x:valid,y:_y})
			summary_writer.add_summary(summary_valid_acc,i)
			summary_writer.add_summary(summary_valid_loss,i)

			print('In valid set, acc: %.4f   loss: %.4f'%(valid_anx[0],valid_anx[1]))
		if i%SAVE_INTERVALS==0:
			saver.save(sess,'./save/Model_'+str(i)+'.ckpt')
	print('======Training completed=======')

	
	saver.restore(sess,'./save/Model_9900.ckpt')
	final_acc = 0.0
	for i in range(1,int(test_step)+1):
		print('%d-th testing' %i)
		val,_y = sess.run([test_image_batch,test_label_batch])
		anx = sess.run([accuracy],feed_dict={x:val,y:_y})
		print(anx[0])
		final_acc = final_acc + anx[0]
	print('=======Testing completed======')
	print('the model acc is %.4f' % (final_acc/test_step))

