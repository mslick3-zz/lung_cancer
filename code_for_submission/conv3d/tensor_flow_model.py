import tensorflow as tf
import numpy as np
from Inputs import *
import sys
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
if len(sys.argv) == 8:
    IMAGE_FOLDER = sys.argv[1]
    INPUT_LABELS = sys.argv[2]
    PROCESSED_DIRECTORY = sys.argv[3]
    FEATURE_MULT = int(sys.argv[4])
    LEARNING_RATE = float(sys.argv[5])
    IMG_SIZE_PX = int(sys.argv[6])
    SLICE_COUNT = int(sys.argv[7])
else:
    IMAGE_FOLDER = '/datadrive/data/full_data/stage1'
    INPUT_LABELS = '/datadrive/data/stage1_labels.csv'
    PROCESSED_DIRECTORY = '/datadrive/output/processed_images_1small/'
    FEATURE_MULT=28800
    LEARNING_RATE = 0.01
    IMG_SIZE_PX = 60
    SLICE_COUNT = 5

PROCESSED_IMAGE_BASED_NAME = "processed_patient_scan_{}.npy"
TEST_LABELS = '/datadrive/data/test_labels.csv'

n_classes = 2
batch_size = 10

x = tf.placeholder('float')
y = tf.placeholder('float')
drop_conv = tf.placeholder('float')
drop_hidden = tf.placeholder('float')
keep_rate = 0.8

def load_npy(patient_id):
    return np.load(PROCESSED_DIRECTORY+PROCESSED_IMAGE_BASED_NAME.format(patient_id))

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME')

def maxpool3d(x):
    #                        size of window         movement of window as you slide about
    return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')


def convolutional_neural_network(x):
    #                # 5 x 5 x 5 patches, 1 channel, 32 features to compute.
    weights = {'W_conv1':tf.Variable(tf.random_normal([3,3,3,1,32])),
               #       5 x 5 x 5 patches, 32 channels, 64 features to compute.
               'W_conv2':tf.Variable(tf.random_normal([3,3,3,32,64])),
               #                                  64 features
               'W_fc':tf.Variable(tf.random_normal([FEATURE_MULT,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    #                            image X      image Y        image Z
    x = tf.reshape(x, shape=[-1, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1])

    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool3d(conv1)

    conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool3d(conv2)

    fc = tf.reshape(conv2,[-1, FEATURE_MULT])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    prediction = tf.matmul(fc, weights['out'])+biases['out']
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)
    return prediction, cost, optimizer 


def train_neural_network(x, train_data_x, train_data_y, validation_data_x, validation_data_y, test_data):
    prediction, cost, optimizer = convolutional_neural_network(x)
    saver = tf.train.Saver()
    hm_epochs = 100
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        successful_runs = 0
        total_runs = 0

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for data, label in zip(train_data_x, train_data_y):
                total_runs += 1
                try:
                    X = data
                    Y = label
                    _, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
                    epoch_loss += c
                    successful_runs += 1
                except Exception as e:
                    # I am passing for the sake of notebook space, but we are getting 1 shaping issue from one
                    # input tensor. Not sure why, will have to look into it. Guessing it's
                    # one of the depths that doesn't come to 20.
                    pass

            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
            a = tf.argmax(prediction, 1)
            b = tf.argmax(y, 1)
            correct = tf.equal(a, b)
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy:',accuracy.eval({x:[i for i in validation_data_x], y:[i for i in validation_data_y]}))
            Y_pred = tf.nn.softmax(prediction, name='softmax_tensor') #convert to probabilities 
            #pred = tf.argmax(Y_pred,1)
            pred = tf.nn.softmax(prediction)
            #print('starting predictions on test')
            validation_result = [] 
            for idx in range(0, len(validation_data_x)): 
                patient_data = validation_data_x[idx]
                model_eval = Y_pred.eval(feed_dict={x:patient_data}, session = sess) #elt index 1 is cancer 
                validation_result.append(model_eval[0][1]) 
            print(validation_result)
            validation_data_cancer = [i[1] for i in validation_data_y] 
            print(validation_data_cancer) 
            print(log_loss(validation_data_cancer, validation_result))

        print('Done. Finishing accuracy:')
        print('Accuracy:',accuracy.eval({x:[i for i in validation_data_x], y:[i for i in validation_data_y]}))
        print('fitment percent:',successful_runs/total_runs)

def get_data():
	all_patients, train_patients, test_patients = get_patients(IMAGE_FOLDER, INPUT_LABELS, TEST_LABELS)
	train_data_loaded = []
	train_data_loaded = [load_npy(pat) for pat in train_patients.ix[:, 0]]
	test_data = []
	test_data = [load_npy(pat) for pat in test_patients.ix[:, 0]]
	train_data_loaded_label = train_patients.ix[:, 1]
	test_data_labels = test_patients.ix[:,1] 
	train_labels = []
	for label in train_data_loaded_label:
		if label == 1: train_labels.append(np.array([0,1]))
		elif label == 0: train_labels.append(np.array([1,0]))
	test_labels = []
	for label in test_data_labels:
		if label == 1: test_labels.append(np.array([0,1]))
		elif label == 0: test_labels.append(np.array([1,0]))
	print("train_data,img_size,{},labels_size,{}".format(len(train_data_loaded), len(train_labels)))
	print("test_data,img_size,{},labels_size,{}".format(len(test_data), len(test_labels)))
	return train_data_loaded, train_labels, test_data, test_labels

def run():
	train_data_x, train_data_y, validation_data_x, validation_data_y  = get_data()
	train_neural_network(x, train_data_x, train_data_y, validation_data_x, validation_data_y, [])

if __name__ == '__main__':
    """
    Run tensorflow from processed_images_tutorial
    """
    run()

