# http://news.91.com/mip/s597361a675eb.html
'''
使用Python+Tensorflow的CNN技术快速识别验证码
'''
# STEP1 图片处理
from PIL import Image

def initTable():		# 二值化，只保留像素122的字母
	table = []
	for i in range(256):
		if i < 123 and i > 121:
			table.append(0)
		else:
			table.append(1)
	return table

for i in range(1,11):
	im = Image.open('F:/capthca2/' + str(i) + '.png')
	im = im.convert('L')	# 转成灰度图片		  im.convert() ??
	binaryImage = im.point(initTable(), '1')	# im.point() ??
	region = (20, 36, 470, 150)
	img = binaryImage.crop(region)
	img.show()

# 验证码信息 像素114*450 最大6个字母 每个字母由26个0或1表示
IMAGE_HEIGHT = 114
IMAGE_WIDTH = 450
MAX_CAPTCHA = 6
CHAR_SET_LEN = 26

# 随机从训练集中提取验证码图片
# 训练集已由名字进行了标签
#if 训练模型
def get_name_and_image():
	all_image = os.listdir('F:/captcha4/')	# 获取所有文件 文件名组成的列表
	random_file = random.randint(0,3429)	# 获得任意数
	base = os.path.basename('F:/captcha4/' + all_image[random_file])	# 某一个文件
	name = os.path.splitext(base)[0]		# 分割扩展名
	image = Image.open('F:/captcha4/' + all_image[random_file])			# 打开某一文件
	image = np.array(image)					# 图片的像素文件
	return name, image
#else 预测模型
def get_name_and_image():
	all_image = os.listdir('F:/captcha5/')	# 获取所有文件 文件名组成的列表
	random_file = random.randint(0,9)	# 获得任意数
	base = os.path.basename('F:/captcha5/' + all_image[random_file])	# 某一个文件
	name = os.path.splitext(base)[0]		# 分割扩展名
	image = Image.open('F:/captcha5/' + all_image[random_file])			# 打开某一文件
	image = np.array(image)					# 图片的像素文件
	return name, image


# 名字转换成向量
'''
	x: 00000000000000000000000100
	u: 00000000000000000000100000
	k: 00000000001000000000000000
	a: 10000000000000000000000000
	n: 00000000000001000000000000
	g: 00000010000000000000000000
'''
def name2vec(name):
	vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)	# 1行 6*26列
	for i, c in enumerate(name):
		idx = i * 26 + ord(c) - 97
		vector[idx] = 1
	return vector

# 向量转换成名字
def vec2name(vec):
	name = []
	for i in vec:
		a = chr(i+97)
		name.append(a)
	return "".join(name)

# 生成一个训练batch
# 默认一次采集64张验证码作为一次训练
# 注意：get_name_and_image获得的image是一个布尔值的矩阵
def get_next_batch(batch_size = 64):
	batch_x = np.zeros([batch_size, IMAGE_HEIGHT*IMAGE_WIDTH])	# 64行114*450列
	batch_y = np.zeros([batch_size, MAX_CAPTCHA*CHAR_SET_LEN])	# 64行6*26列

	for i in range(batch_size):
		name, image = get_name_and_image()
		batch_x[i, :] = 1*(image.flatten())
		batch_y[i, :] = name2vec(name)
	return batch_x, batch_y

# 采用3个卷积层和1个全连接层
# 每个卷积层中都选用2*2的最大池化层和dropout层，卷积核尺寸5*5
# 114*450的图片经过3层池化层，长宽都压缩了8倍，得到15*57大小
# 定义CNN
def crack_captcha_cnn(w_alpha = 0.01, b_alpha = 0.1):
	x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH])
	# 3 conv layer
	w_cl = tf.Variable(w_alpha * tf.random_normal([5, 5, 1, 32]))
	b_cl = tf.Variable(b_alpha * tf.random_normal([32]))
	conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_cl, strides=[1, 1, 1, 1], padding='SAME'), b_cl))
	conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	conv1 = tf.nn.dropout(conv1, keep_prob)

	w_c2 = tf.Variable(w_alpha * tf.random_normal([5, 5, 32, 64]))
	b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
	conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
	conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	conv2 = tf.nn.dropout(conv2, keep_prob)

	w_c3 = tf.Variable(w_alpha * tf.random_normal([5, 5, 64, 64]))
	b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
	conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
	conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	conv3 = tf.nn.dropout(conv3, keep_prob)

	# Fully connected layer
	w_d = tf.Variable(w_alpha * tf.random_normal([15 * 57 * 64, 1024]))
	b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
	dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
	dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
	dense = tf.nn.dropout(dense, keep_prob)

	w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
	b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
	out = tf.add(tf.matmul(dense, w_out), b_out)
	return out

# 开始训练 使用sigmoid_cross_entropy_with_logits()交叉熵来比较loss
# 用adam优化I器来优化。
# 输出每一步的loss值，每一百步输出一次准确率。
# 目标：当准确率达99%后，结束训练。
# keep_prob=0.5
def train_crack_captcha_cnn():
	output = crack_captcha_cnn()
	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labes=Y))
	optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

	predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
	max_idx_p = tf.argmax(predict, 2)
	max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
	correct_pred = tf.equal(max_idx_p, max_idx_l)
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		step = 0
		while true:
			batch_x, batch_y = get_next_batch(64)
			_, loss_ = sess.run([optimizer, loss], feed_dict={X:batch_x, Y:batch_y, keep_prob:0.5})
			print(step, loss_)

			# 每100step计算一次准确率
			if step % 100 == 0:
				batch_x_test,batch_y_test = get_next_batch(100)
				acc = sess.run(accuracy, feed_dict={X:batch_x_test, Y:batch_y_test, keep_prob:1.})
				print(step, acc)
				#如果准确率大于99%，保存模型，完成训练
				if acc > 0.99:
					saver.save(sess, './crack_capcha.model', global_step=step)
					break
			step += 1

# 模型训练
train_crack_captcha_cnn()

# 预测
# ※需要注释掉上一句
def crack_captcha():
	output = crack_captcha_cnn()
	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess, tf.train.latest_checkpoint('.'))
		n = 1
		while n <= 10:
			text, image = get_name_and_image()
			image = 1 * (image.flatten())
			predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA,CHAR_SET_LEN]), 2)
			text_list = sess.run(predict, feed_dict={X:[image], keep_prob: 1})
			vec = text_list[0].tolist()
			predict_text = vec2name(vec)
			print("正确：{} 预测：{}".format(text, predict))
			n += 1
crack_captcha()
