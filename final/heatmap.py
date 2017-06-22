import numpy as np
import pandas as pd
import random as rand
import matplotlib.pyplot as plt
from termcolor import colored,cprint
### keras
from keras import backend as K
from keras.models import load_model

################# path ##################
train_path = './data/train.csv'
test_path = './data/test.csv'
out_path = './data/sub.csv'
model_path = './train.h5'

############### parameter ###############
nb_epoch = 10
batch_size = 128
split_ratio = 0.25

#########################################
#                Main                   #
#########################################
def main():
    ### parsing
    model = load_model(model_path)

    train = pd.read_csv(train_path)
    train_label = train['label']
    train_label = np.array(train_label)
    train = np.array(train)
    train = np.delete(train, 0, 1)
    train = train / 255
    train = train.reshape((train.shape[0], 28, 28, 1))

    input_img = model.input
    img_ids = list(range(10))

    for idx in img_ids:
        val_proba = model.predict(train[idx].reshape(1, 28, 28, 1))
        pred = val_proba.argmax(axis=-1)
        target = K.mean(model.output[:, pred])
        grads = K.gradients(target, input_img)[0]
        fn = K.function([input_img, K.learning_phase()], [grads])

        ### heatmap processing
        heatmap = fn([train[idx].reshape(1, 28, 28 ,1), 0])[0]
        heatmap_mean = np.mean(heatmap.reshape(28, 28))
        heatmap_std = np.std(heatmap.reshape(28, 28))
        heatmap = ((heatmap.reshape(28, 28) - heatmap_mean) / heatmap_std).reshape(1, 28 ,28, 1)
        
        threshold = 0.4
        see = train[idx].reshape(28, 28)
        for i in range(28):
            for j in range(28):
                    if heatmap.reshape(28, 28)[i, j] <= threshold:
                        see[i, j ] = np.mean(see)

        
        plt.figure()
        plt.imshow(heatmap.reshape(28, 28), cmap=plt.cm.jet)
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig("./heatmap/cmap_"+ str(idx) +".png", dpi=100, format = 'png')

        plt.figure()
        plt.imshow(see,cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig("./heatmap/partial_see_" + str(idx) + ".png", dpi=100, format = 'png')


if __name__ == '__main__':
    main()
