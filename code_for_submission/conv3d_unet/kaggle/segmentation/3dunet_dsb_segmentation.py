import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import cv2
import zarr
import glob

from ConfigParser import SafeConfigParser
from keras.models import load_model
from keras import backend as K

def load_zarr(patient_id):
    lung_mask_zarr = zarr_load_group['lung_mask'][patient_id]
    return np.array(lung_mask_zarr).astype('float32')

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def save_zarr(id_patient, cand):
    cand_group.array(id_patient, cand, 
            chunks=(1, 1, 17, 21, 21), compressor=zarr.Blosc(clevel=9, cname="zstd", shuffle=2),
            synchronizer=zarr.ThreadSynchronizer())
    return

def load_unet(check_name):
    check_model = luna_unet_model + '/{}.h5'.format(check_name)
    model = load_model(check_model, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    return model

def predict_dsb3d(ids):
    model = load_unet('3DUNet_genfulldata')
    masks = np.full((1, 1, 136, 168, 168), 7.).astype('float32')
    for i in range(len(ids)):
        masks = np.full((1, 1, 136, 168, 168), 7.).astype('float32')
        patient_id = ids[i]
        print(patient_id)
        mask = load_zarr(ids[i])
        mask = mask.swapaxes(1, 0)
        num_slices = mask.shape[1]
        offset = (136 - num_slices)
        if offset == 0:
            masks[0, :, :, :, :] = mask[:, :, :, :]
        if offset > 0:
            begin_offset = int(np.round(offset/2))
            end_offset = int(offset - begin_offset)
            masks[0, :, begin_offset:-end_offset, :, :] = mask[:, :, :, :]
        if offset < 0:
            offset = -(136 - num_slices)
            begin_offset = int(np.round(offset/2))
            end_offset = int(offset - begin_offset)
            masks[0, :, :, :, :] = mask[:, begin_offset:-end_offset, :, :]
        preds = (model.predict(masks, batch_size = 1)).astype('float32')
        thres = 0.9
        preds[preds <= thres] = 0.
        preds[preds > thres] = 1.
        cands = masks * preds
        print(cands.shape) 
        save_zarr(patient_id, cands)
    return

config = SafeConfigParser()
config.read('../../file_paths.config')
processed_img_dir = config.get('main', 'kaggle_processed_image_output')
kaggle_processed_images_store = zarr.DirectoryStore(processed_img_dir)
zarr_group = zarr.hierarchy.open_group(store=kaggle_processed_images_store, mode='a')
lung_mask_group = zarr_group.require_group('lung_mask')
cand_group = zarr_group.require_group('candidates')

zarr_load_group = zarr.hierarchy.open_group(store=kaggle_processed_images_store, mode='r')

dsbpats = [x for x in os.listdir(processed_img_dir + '/lung_mask') if not x[0] == '.']
dsbpats.sort()
print(dsbpats)
luna_unet_model=config.get('main', 'luna_unet_model')
predict_dsb3d(dsbpats)
