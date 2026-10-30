import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
from main import prediction

#opening the training labels array and assigning it to the variable Y which tells the AI whether it is correct, has a size of (1, 1500)
lp = open("test_labels", "rb")
test_Y = pickle.load(lp)
lp.close()

#opening the training set array and assigning it to the variable X which has the data that trains the AI (the pictures), has a size of (1500, 100, 100, 3)
lp = open("test_set", "rb")
test_X = pickle.load(lp)
lp.close()

lp = open("params_test", "rb")
params = pickle.load(lp)
lp.close()

if __name__ == "__main__":
    test_X_flattened = test_X.reshape(test_X.shape[0], -1).T
    pred = prediction(test_X_flattened, test_Y, params)
    for i in range(len(test_X)):
        print("Predicted:", pred[0, i], "  Actual:", test_Y[0, i], "\n")
        plt.imshow(test_X[i])
        plt.show()
    print(pred)
    print(test_Y)

