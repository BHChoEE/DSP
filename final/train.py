import numpy as np
import pandas as pd
### keras

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adamax
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical

################# path ##################
train_path = './data/train.csv'
test_path = './data/test.csv'
out_path = './data/sub.csv'

############### parameter ###############
nb_epoch = 10
batch_size = 128
split_ratio = 0.25

#########################################
#                Util                   #
#########################################
def get_cnnmodel():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation = 'relu', input_shape = (28, 28, 1) ))
    model.add(Conv2D(64, (3, 3), activation = 'relu') )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2) ) )
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), activation = 'relu') )
    model.add(Conv2D(128, (3, 3), activation = 'relu') )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2) ) )
    model.add(Dropout(0.3))
    
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = 'softmax'))

    adamax = Adamax(lr = 0.008, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 0.0)
    model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy', ], optimizer = "adamax")
    model.summary()

    return model


#########################################
#                Main                   #
#########################################
def main():
    ### Parsing
    train = pd.read_csv(train_path)
    train_label = train['label']
    train_label = np.array(train_label)
    train_label = to_categorical(train_label, num_classes = 10)
    train = np.array(train)
    train = np.delete(train, 0, 1)
    train = train / 255
    train = train.reshape((train.shape[0], 28, 28, 1))
    '''
    test = pd.read_csv(test_path)
    test = np.array(test)
    test = test / 255
    test = test.reshape((test.shape[0], 28, 28, 1))
    '''
    ### model 
    model = get_cnnmodel()

    ### fitting
    earlystopping = EarlyStopping(
        monitor = 'val_acc',
        patience = 5,
        verbose = 1,
        mode = 'max'
    )
    checkpoint = ModelCheckpoint(
        filepath = 'best.hdf5',
        save_best_only = True,
        save_weights_only = True,
        monitor = 'val_acc',
        verbose = 1,
        mode = 'max'
    )
    model.fit(
        train, train_label,
        validation_split = split_ratio,
        epochs = nb_epoch,
        batch_size = batch_size,
        callbacks = [earlystopping, checkpoint]
    )
    model.save('train.h5')
    
if __name__ == '__main__':
    main()