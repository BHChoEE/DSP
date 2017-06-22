import numpy as np
import pandas as pd
import random as rand
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
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
#                Util                   #
#########################################
def plot_confusion_matrix(cm , classes, title = 'Confusion matrix', cmap = plt.cm.jet):
    cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

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

    ### get validation data
    choose_idx = rand.sample(range(0, train.shape[0]-1), int(train.shape[0] * split_ratio))
    val_train = train[choose_idx]
    val_train_label = train_label[choose_idx]
    ### get confusion matrix
    pred = model.predict_classes(val_train)
    conf_mat = confusion_matrix(val_train_label, pred)
    ### plot
    plt.figure()
    plot_confusion_matrix(conf_mat, classes=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    plt.savefig("conf_mat.png", dpi = 300, format = "png")

if __name__ == '__main__':
    main()