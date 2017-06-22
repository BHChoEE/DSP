import numpy as np
import pandas as pd
### keras
from keras.models import load_model

################# path ##################
train_path = './data/train.csv'
test_path = './data/test.csv'
out_path = './data/sub.csv'


def main():
    test = pd.read_csv(test_path)
    test = np.array(test)
    test = test / 255
    test = test.reshape((test.shape[0], 28, 28, 1))

    model = load_model('train.h5')
    score = model.predict(test, batch_size = 32, verbose = 0)
    #print(score.shape)

    #output the score into csv
    with open(out_path,'w') as ofile:
        ofile.write("ImageId,Label\n")
        for i in range(score.shape[0]):
            ofile.write(str(i+1))
            ofile.write(",")
            out = np.argmax(score[i])
            ofile.write(str(out))
            ofile.write('\n')
    

if __name__ == '__main__':
    main()