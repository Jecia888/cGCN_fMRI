import os
import h5py
import numpy as np
import keras
import pandas as pd
from keras import backend as k
from keras import optimizers
from time import gmtime, strftime
from model import *
from matplotlib import pyplot
import matplotlib.pyplot as plt

ROI_N = 236
frames = 100
# ########################################## Load data ########################################
# Download the data from https://drive.google.com/file/d/1l029ZuOIUY5gehBZCAyHaJqMNuxRHTFc/view?usp=sharing
with h5py.File('HCP.h5', 'r') as f:
    print(list(f.keys()))
    x_train, x_val, x_test = f['x_train'][()], f['x_val'][()], f['x_test'][()]
    y_train, y_val, y_test = f['y_train'][()], f['y_val'][()], f['y_test'][()]


x_train = np.expand_dims(x_train, -1) # (200, 100, 236, 1)
x_val = np.expand_dims(x_val, -1)
x_test = np.expand_dims(x_test, -1)
print (x_train.shape)
print (x_val.shape)
print (x_test.shape)


# Convert class vectors to binary class matrices.
num_classes = 100
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print (y_train.shape)
print (y_val.shape)
print (y_test.shape)

################################ Set parameter ###############################
print()

k = 5 # (3, 5, 10, 20)
batch_size = 8
epochs = 100
l2_reg = 1e-4
dp = 0.5
lr = 1e-5

print('dp:', dp)
print('l2:', l2_reg)
print('batch_size:', batch_size)
print('epochs:', epochs)
print('lr:', lr)


ROI_N = 236
random_FC = np.random.rand(ROI_N, ROI_N)
random_FC[np.diag_indices(ROI_N)] = 1
np.save('FC_random', random_FC)
plt.imshow(random_FC)
plt.show()

file_name='avg_edgeconv_k_' + str(k) + '_l2_' + str(l2_reg) + '_dp_' + str(dp)
print('file_name:', file_name)

os.system('mkdir tmp') # folder for the trained model
tmp_name = 'tmp/tmp_' + file_name + '_' + strftime("%Y_%m_%d_%H_%M_%S", gmtime()) + '.keras'
print('output tmp name:', tmp_name)

print("hi1")
############################################### Get pre-trained model  ############################
weight_name = None

# # Find and load best pre-trained model
# weight_path = 'tmp/%s/'%(site)
# all_weights = os.listdir(weight_path)
# all_right_models = {}
# for n in all_weights:
#     if '.hdf5' in n:
#         n_split = n.split('_')
#         if int(n_split[1+n_split.index('k')]) == k:
#         # if int(n_split[1+n_split.index('k')]) == k and \
#         #     float(n_split[1+n_split.index('l2')]) == l2_reg:
#             all_right_models[float(n_split[1+n_split.index('valAcc')])] = n

# if all_right_models:
#     best_acc = np.max(list(all_right_models.keys()))
#     print('-------best acc %f, model name: %s'%(best_acc, all_right_models[best_acc]))
#     weight_name = weight_path+all_right_models[best_acc]

################################ get model  ######################################################
# Download 'FC.npy' from https://drive.google.com/file/d/1WP4_9bps-NbX6GNBnhFu8itV3y1jriJL/view?usp=sharing

print("hi2")
model = get_model(
    graph_path='FC_random.npy', 
    ROI_N=ROI_N,
    frames=frames,
    kernels=[8,8,8,16,32,32], 
    k=k, 
    l2_reg=l2_reg, 
    dp=dp,
    num_classes=num_classes, 
    weight_path=weight_name, 
    skip=[0,0])
model.summary()


######################################## Training ####################################################
model.compile(loss=['categorical_crossentropy'], 
              optimizer=optimizers.Adam(learning_rate=lr),
              metrics=['accuracy'])

print('Train...')
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.5,
                                              patience=10, min_lr=1e-6)
lr_hist = []
class Lr_record(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        tmp = K.eval(model.optimizer.learning_rate)
        lr_hist.append(tmp)
        print('Ir:', tmp)
lr_record = Lr_record()
earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

checkpointer = keras.callbacks.ModelCheckpoint(monitor='val_acc', filepath=tmp_name, 
                                                verbose=1, save_best_only=True)
model_history = model.fit(x_train, y_train,
                            shuffle=True,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(x_val, y_val),
                            callbacks=[checkpointer, lr_record, reduce_lr, earlystop])


'''
print('validation...')
with open(model_path, 'rb') as fp:
    tmp = imp.load_module(model_path[:-3], fp, model_path,('.py', 'rb', imp.PY_SOURCE))
    
model_best = get_model(
    graph_path='FC.npy', 
    ROI_N=ROI_N,
    frames=frames,    
    kernels=[8,8,8,16,32,32], 
    k=k,
    l2_reg=l2_reg, 
    num_classes=num_classes, 
    weight_path=tmp_name, 
    skip=[0,0])

model_best.compile(loss=['categorical_crossentropy'], 
              optimizer=optimizers.Adam(lr=lr),
              metrics=['accuracy'])

val_tmp = model_best.evaluate(x=x_val, y=y_val, batch_size=batch_size,verbose=1)
print('validation:', val_tmp)

test_tmp = model_best.evaluate(x=x_test, y=y_test, batch_size=batch_size,verbose=1)
print('test:', test_tmp)

######################################## save log and model #######################################

save_logs_models(model, model_history, acc=val_tmp[1],
    folder='tmp/', 
    lr_hist=lr_hist, file_name=file_name, loss_name='loss', 
    acc_name='acc', tmp_name=tmp_name)
'''