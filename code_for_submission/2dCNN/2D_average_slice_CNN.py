import tensorflow as tf
import numpy as np
import sys
import os
import pandas as pd
from sklearn.cross_validation import train_test_split

if len(sys.argv) == 4:
    IMAGE_FOLDER = sys.argv[1]
    INPUT_LABELS = sys.argv[2]
    PROCESSED_DIRECTORY = sys.argv[3]
else:
    IMAGE_FOLDER = '/datadrive/data/full_data/stage1'
    INPUT_LABELS = '/datadrive/data/stage1_labels.csv'
    PROCESSED_DIRECTORY = '/datadrive/output/processed_images_3/'
    #'/datadrive/project_code/cs6250_group_project/processed_images_tutorial/'
    
PROCESSED_IMAGE_BASED_NAME = "processed_patient_scan_{}.npy"
    
def load_npy(patient_id):
    return np.load(PROCESSED_DIRECTORY+PROCESSED_IMAGE_BASED_NAME.format(patient_id))

def get_patients(image_directory, labels_path):
    """
    Load list of train and test patients. For train patients, return outcome label along with patient id
    :param image_directory: directory containing images
    :param labels_path: path to training dataset labels
    :return: 2 pandas dataframes, each contains id and cancer indicator
                1st contains train patients [patient id and cancer indicator (0/1)]
                2nd contains test patients [patient id and cancer indicator is null]
    """
    patients = os.listdir(image_directory)
    train_labels = pd.read_csv(labels_path)
    patients_df = pd.DataFrame({'id': patients})
    patients_df = pd.merge(patients_df, train_labels, how='left', on='id')
    patients_df = patients_df.reindex(np.random.permutation(patients_df.index))
    train_patients = patients_df[pd.notnull(patients_df['cancer'])]
    test_patients = patients_df[pd.isnull(patients_df['cancer'])]
    return patients_df, train_patients, test_patients
        
def get_data():
    all_patients, train_patients, test_patients = get_patients(IMAGE_FOLDER, INPUT_LABELS)
    train_data_loaded = []
    train_data_loaded = [load_npy(pat) for pat in train_patients.ix[:, 0]]
    train_data_loaded_label = train_patients.ix[:, 1]
    train_labels = []
    for label in train_data_loaded_label:
        if label == 1: train_labels.append(np.array([0,1]))
        elif label == 0: train_labels.append(np.array([1,0]))
    much_data = []
    for indx in range(0,len(train_data_loaded)):
        much_data.append([train_data_loaded[indx], train_labels[indx]]) 
    
    train_data = much_data[:-100]
    validation_data = much_data[-100:]     
    validation_labels = train_patients[-100:]
    
    return train_data, validation_data, validation_labels

x = tf.placeholder('float')
y = tf.placeholder('float')

IMG_SIZE_PX = 120
LEARNING_RATE = .01
hm_epochs = 30
n_classes = 2
batch_size = 15
keep_rate = 0.3

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    #                        size of window         movement of window as you slide about
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def convolutional_neural_network(x):
    # Equal to size of matrix where the original pixel size is halved by each max pool layer
    # e.g. 120 px size -> 3 max pool layers w/ 2x2 patch sizes goes to: 120/2/2/2 = 15.
    # New dimensions are multiplied by the number of features to generate. e.g. 128 features
    # Calculation is new dimesnsions by features: (15 * 15) * 128
    NODES = 8 * 8 * 256
    
    #                # 5 x 5 x 5 patches, 1 channel, 32 features to compute.
    weights = {'W_conv1':tf.Variable(tf.random_normal([3,3,1,32])),
               #       5 x 5 x 5 patches, 32 channels, 64 features to compute.
               'W_conv2':tf.Variable(tf.random_normal([3,3,32,64])),
               #                                  64 features
               'W_conv3':tf.Variable(tf.random_normal([3,3,64,128])),
               'W_conv4':tf.Variable(tf.random_normal([3,3,128,256])),
               'W_fc':tf.Variable(tf.random_normal([NODES, 1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_conv3':tf.Variable(tf.random_normal([128])),
               'b_conv4':tf.Variable(tf.random_normal([256])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    # [(-1 means dynamic batch size)   image X   image Y   Color channels]
    x = tf.reshape(x, shape=[-1, IMG_SIZE_PX, IMG_SIZE_PX, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)
    
    conv3 = tf.nn.relu(conv2d(conv2, weights['W_conv3']) + biases['b_conv3'])
    conv3 = maxpool2d(conv3)
    
    conv4 = tf.nn.relu(conv2d(conv3, weights['W_conv4']) + biases['b_conv4'])
    conv4 = maxpool2d(conv4)

    fc = tf.reshape(conv4, [-1, NODES])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output


def train_neural_network(x, x_train, x_test, y_train, y_test):
    NNpreds = []
    NNprobs = []

    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        
        successful_runs = 0
        total_runs = 0

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for data in zip(x_train, y_train):
                total_runs += 1
                X = data[0]
                Y = data[1]
                _, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
                epoch_loss += c
                
            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
            a = tf.argmax(prediction, 1)
            b = tf.argmax(y, 1)
            correct = tf.equal(a, b)
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy:',accuracy.eval(feed_dict={x: x_test,y: y_test}))

        print('Done. Finishing accuracy:')
        print('Accuracy:',accuracy.eval(feed_dict={x: x_test,y: y_test}))
        
        pred = tf.argmax(y,1)
        NNpreds.extend(prediction.eval(feed_dict={x: x_test,y: y_test}))
        
        probs = y/tf.reduce_sum(y, 0)
        NNprobs.extend(probs.eval(feed_dict={x: x_test,y: y_test}))
        
        return NNprobs
        
        
def run():
    train, validation, validation_ids = get_data()
    ids = validation_ids.ix[:, 0]
    
    x_train = [i[0] for i in train]
    y_train = [i[1] for i in train]
    x_test = [i[0] for i in validation]
    y_test = [i[1] for i in validation]
    
    NNprobs = np.zeros((len(validation), 1))    
    
    probs = train_neural_network(x, x_train, x_test, y_train, y_test)
    
    NNprobs[:, i] = np.array(probs)[:,1]
    predictions = pd.DataFrame(NNpreds, columns = ["slice"])
    predictions[predictions['slice'] > 0] = 1

    predictions.to_csv('/datadrive/output/avg_preds.csv', header=True, index=False)

if __name__ == '__main__':
    run()
