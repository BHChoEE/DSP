import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
### keras
from keras.models import load_model
from keras import backend as K

################# path ##################
train_path = './data/train.csv'
test_path = './data/test.csv'
out_path = './data/sub.csv'
model_path = './train.h5'

############### parameter ###############
nb_epoch = 10
batch_size = 128
split_ratio = 0.25
NUM_STEPS = 20
RECORD_FREQ = 5
nb_filter = 64
step = 1

#########################################
#                Util                   #
#########################################
def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-7)

def grad_ascent(input_image_data,iter_func):
    """
    Implement this function!
    

    step = 1
    loss_value, grads_value = iter_func([input_image_data])
    input_img_data += grads_value * step
    """
    return input_img_data

#########################################
#                Main                   #
#########################################
def main():
    ### parsing
    model = load_model(model_path)
    layer_dict = dict([layer.name, layer] for layer in model.layers[0:])
    input_img = model.input
    name_ls = ['conv2d_1']
    collect_layers = [ layer_dict[name].output for name in name_ls ]
    
    for cnt, c in enumerate(collect_layers):
        filter_imgs = [[] for i in range(NUM_STEPS//RECORD_FREQ)]
        for filter_idx in range(nb_filter):
            # Start from a gray image with some random noise
            input_img_data = np.random.random((1, 28, 28, 1)) # random noise
            target = K.mean(c[:, :, :, filter_idx])
            grads = normalize(K.gradients(target, input_img)[0])
            iterate = K.function([input_img], [target, grads])
            
            for i in range(NUM_STEPS):
                loss_value, grads_value = iterate([input_img_data])
                input_img_data += grads_value * step
                #record the temporary img to filter_imgs
                if i % RECORD_FREQ == 0:
                    filter_imgs[int(i / RECORD_FREQ)].append([input_img_data.reshape(28, 28).tolist()])

        for it in range(NUM_STEPS//RECORD_FREQ):
            fig = plt.figure(figsize=(14, 8))
            for i in range(nb_filter):
                ax = fig.add_subplot(nb_filter/16, 16, i+1)
                ax.imshow(filter_imgs[it][i][0], cmap='BuGn')
                plt.xticks(np.array([]))
                plt.yticks(np.array([]))
                plt.tight_layout()
            fig.suptitle('Filters of layer {} (# Ascent Epoch {} )'.format(name_ls[cnt], it*RECORD_FREQ))
            fig.savefig("./hack_filter/Filter"+str(i)+"Epoch"+str(it)+".png" ,format = 'png')


if __name__ == '__main__':
    main()
