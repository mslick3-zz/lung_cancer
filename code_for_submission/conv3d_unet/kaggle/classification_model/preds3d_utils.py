import os
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import zarr
import glob
import matplotlib.pyplot as plt
import os
import glob
import time
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report,accuracy_score,log_loss
from ConfigParser import SafeConfigParser

config = SafeConfigParser()
config.read('../../file_paths.config')
zarr_dir = config.get('main', 'kaggle_processed_image_output')
zarr_store = zarr.DirectoryStore(zarr_dir)
zarr_load_group = zarr.hierarchy.open_group(store=zarr_store, mode='r')
dsb_pats = os.listdir(zarr_dir + '/candidates/')
dsb_labels = config.get('main', 'kaggle_dataset_labels')
classification_ckpt = config.get('main', 'kaggle_classificaton_model_store')


def generate_train(start, end, seed = None):
    df = pd.read_csv(dsb_labels)[start:end]
    while True:
        labels = to_categorical(df['cancer'])
        for i in range(len(df)):
            cand = load_zarr('{}'.format(df.iloc[i, 0]))
            cand = cand/255.
            y = labels[[i]]
            yield(cand, y)
            
def generate_val(start, end, seed = None):
    df = pd.read_csv(dsb_labels)[start:end]
    while True:
        labels = to_categorical(df['cancer'])
        for i in range(len(df)):
            cand = load_zarr('{}'.format(df.iloc[i, 0]))
            cand = cand/255.
            y = labels[[i]]
            yield(cand, y)

def generate_test(start, end, seed = None):
    df = pd.read_csv(dsb_labels)[start:end]
    labels = to_categorical(df['cancer'])
    for i in range(len(df)):
        cand = load_zarr('{}'.format(df.iloc[i, 0]))
        cand = cand/255.
        y = labels[[i]]
        yield(cand, y)

def load_zarr(patient_id):
    lung_cand_zarr = zarr_load_group['candidates'][patient_id]
    return np.array(lung_cand_zarr).astype('float32')


def cnn3d_genfit(name, nn_model, epochs, start_t, end_t, start_v, end_v, start_test, end_test, nb_train, nb_val, check_name = None):
    callbacks = [EarlyStopping(monitor='val_loss', patience = 15, 
                                   verbose = 1),
    ModelCheckpoint(classification_ckpt + '/{}.h5'.format(name), 
                        monitor='val_loss', 
                        verbose = 0, save_best_only = True)]
    if check_name is not None:
        check_model = classification_ckpt + '/{}.h5'.format(check_name)
        model = load_model(check_model)
    else:
        model = nn_model
    model.fit_generator(generate_train(start_t, end_t), nb_epoch = epochs, verbose = 1, 
                        validation_data = generate_val(start_v, end_v), 
                        callbacks = callbacks,
                        samples_per_epoch = nb_train, nb_val_samples = nb_val)
    test_pred = [] 
    test_actual = [] 
    for img, label in generate_test(start_test, end_test): 
        test_pred.append(model.predict(img)[0][1])
        test_actual.append(label[0][1]) 
    print(test_pred)
    print(test_actual) 
    printMetrics(test_pred, test_actual) 
    return

def printMetrics(test_pred, test_actual): 
    test_pred_labels = np.rint(test_pred) 
    
    #print roc 
    fpr, tpr, _ = roc_curve(test_actual, test_pred)
    roc_area = auc(fpr, tpr) 
    print("roc_area,{}".format(roc_area))
    #plt.figure()
    #lw = 2
    #plt.plot(fpr, tpr, color='darkorange',
    #     lw=lw, label='ROC curve (area = %0.2f)' % roc_area)
    #plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.05])
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')
    #plt.title('Receiver operating characteristic example')
    #plt.legend(loc="lower right")
    #plt.show()
    
    #print precision and recall 
    print(classification_report(test_actual, test_pred_labels))

    #print accuracy 
    print("accuracy,{}".format(accuracy_score(test_actual, test_pred_labels)))

    #print logloss
    print("log_loss,{}".format(log_loss(test_actual, test_pred)))
