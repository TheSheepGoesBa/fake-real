import numpy
import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt
import math
import pickle

#opening the training labels array and assigning it to the variable Y which tells the AI whether it is correct, has a size of (1, 1500)
lp = open("weights/train_labels", "rb")
Y = pickle.load(lp)
lp.close()

#opening the training set array and assigning it to the variable X which has the data that trains the AI on the pictures, has a size of (1500, 100, 100, 3)
lp = open("weights/training_set2", "rb")
X = pickle.load(lp)
lp.close()

#opening the training labels array and assigning it to the variable test_Y which tells the AI whether it is correct, has a size of (1, 100)
lp = open("weights/test_labels", "rb")
test_Y = pickle.load(lp)
lp.close()

#opening the training set array and assigning it to the variable test_X which has the data that tests the AI on the pictures, has a size of (1500, 100, 100, 3)
lp = open("weights/test_set", "rb")
test_X = pickle.load(lp)
lp.close()


#creating the training set and labels from the images
'''
lp = open("fake", "rb")
fakes = pickle.load(lp)
lp.close()

lp = open("real", "rb")
reals = pickle.load(lp)
lp.close()


fakes = []
reals = []
#getting the filename of each picture
for filename in glob.glob('real_and_fake_face/training_fake/*.jpg'):
    #loading it
    orig = Image.open(filename)
    new = orig.resize((100, 100))
    #putting the data into an array of pixels
    im = np.array(new)
    fakes.append(im)
for filename2 in glob.glob('real_and_fake_face/training_real/*.jpg'):
    orig = Image.open(filename2)
    new = orig.resize((100, 100))
    im = np.array(new)
    reals.append(im)
array2 = np.zeros((len(fakes), 100, 100, 3))
array3 = np.zeros((len(reals), 100, 100, 3))
for i in range(len(fakes)):
    array2[i] = fakes[i]
for i in range(len(reals)):
    array3[i] = reals[i]

sp = open("fake2", "wb")
pickle.dump(array2, sp)
sp = open("real2", "wb")
pickle.dump(array3, sp)


train_Y = np.zeros((1, 1500))
test_Y = np.zeros((1, 100))

lp = open("fake1", "rb")
fakes = pickle.load(lp)
lp.close()

lp = open("real1", "rb")
reals = pickle.load(lp)
lp.close()

print(len(reals))
print(len(fakes))
training_set = np.zeros((1500, 100, 100, 3))
test_set = np.zeros((100, 100, 100, 3))
r_count = 0
f_count = 0
for i in range(1500):
    if np.random.randint(0, 2) == 0:
        training_set[i] = reals[r_count]/255
        train_Y[0][i] = 0
        r_count+=1
    else:
        training_set[i] = fakes[f_count]/255
        train_Y[0][i] = 1
        f_count+=1
for i in range(100):
    if np.random.randint(0, 2) == 0:
        test_set[i] = reals[r_count]/255
        test_Y[0][i] = 0
        r_count+=1
    else:
        test_set[i] = fakes[f_count]/255
        test_Y[0][i] = 1
        f_count+=1
sp = open("train_labels", "wb")
pickle.dump(train_Y, sp)
sp = open("training_set2", "wb")
pickle.dump(training_set, sp)
sp = open("test_labels", "wb")
pickle.dump(test_Y, sp)
sp = open("test_set", "wb")
pickle.dump(test_set, sp)
'''


def initialize_params(layer_dimensions):
    L = len(layer_dimensions) # number of layers
    params = {}

    # looping through each layer
    for l in range(1, L):
        # initializing the weight matrix with random values for each layer with size (nodes in current layer, input nodes)
        params["W" + str(l)] = np.random.randn(layer_dimensions[l], layer_dimensions[l-1]) * np.sqrt(2/layers[l-1])
        # initializing the bias matrix with zeros for each layer with size (nodes in current layer, 1)
        params["b" + str(l)] = np.zeros((layer_dimensions[l], 1))

    return params


# the sigmoid curve
def sigmoid(Z):
    return 1/(1+np.exp(-Z))


# implements forward propagation
p = 0.5
def L_forward(X, params):
    caches = []
    L = len(params) // 2   #number of layers
    A = X   # A is the variable for after the activation function sigmoid or relu

    #looping over each layer
    for l in range(1, L):
        Aprev = A  #the previous value of A

        Z = np.dot(params["W" + str(l)], Aprev) + params["b" + str(l)]  #the input for the activation function
        A = np.maximum(0, Z)   #relu(activation function)
        #c = (np.random.rand(*A.shape) < p) / (1-p)
        #A *= c

        caches.append((Z, A, params["W" + str(l)], params["b" + str(l)]))   #adding the values of the variables to a cache to be used later

    Z = np.dot(params["W" + str(L)], A) + params["b" + str(L)]
    A = sigmoid(Z)  #using sigmoid(activation function) for the last input
    caches.append((Z, A, params["W" + str(L)], params["b" + str(L)]))  #adding the values of the variables to a cache to be used later
    return A, caches


#function to compute the cost, how wrong the model was in its predictions(the higher the value, the more wrong)
def cost_compute(AL, Y):
    cost = np.multiply(-np.log(AL), Y) + np.multiply(-np.log(1 - AL), 1 - Y)   #the formula for computing cost
    cost = np.sum(cost)  #adding all of the values up since it's a matrix currently
    return cost


#implementing back propagation to adjust the parameters to train the neural network
def L_backward(X, Y, caches):
    m = X.shape[1]  #number of training examples
    (Z1, A1, W1, b1), (Z2, A2, W2, b2), (Z3, A3, W3, b3), (Z4, A4, W4, b4) = caches  #getting the values from the cache earlier

    #Everything below is just formulas for computing the gradient(or derivative) of the loss function with respect to the weights for each layer
    #And also uses dropout to reduce overfitting
    dZ4 = 1. / m * (A4 - Y)
    dW4 = np.dot(dZ4, A3.T)
    db4 = np.sum(dZ4, axis=1, keepdims=True)

    dA3 = np.dot(W4.T, dZ4)
    dZ3 = np.multiply(dA3, np.int64(A3 > 0))
    dW3 = np.dot(dZ3, A2.T)
    db3 = np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = np.dot(dZ2, A1.T)
    db2 = np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = np.dot(dZ1, X.T)
    db1 = np.sum(dZ1, axis=1, keepdims=True)

    #assigning each derivative to a dictionary for use later
    grads = {"dZ4": dZ4, "dW4": dW4, "db4": db4,
             "dA3": dA3,"dZ3": dZ3, "dW3": dW3, "db3": db3,
             "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
             "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return grads


#Function to create random mini batches, which are just small training sets
def mini_batches_rdm(X, Y, size=32):
    m = X.shape[1]  #number of training examples
    mini_batches = []

    #Shuffling X and Y to have a random list every time the function is called
    perm = list(np.random.permutation(m))
    new_X = X[:, perm]
    new_Y = Y[:, perm].reshape((1, m))

    complete_mini_batches = math.floor(m/size)   #number of complete mini batches in the training set
    for k in range(complete_mini_batches):
        #Making mini batches from the shuffled training set and lables using slicing from (start number for the current mini batch : end number)
        mini_batch_X = new_X[:, k * size:(k + 1) * size]
        mini_batch_Y = new_Y[:, k * size:(k + 1) * size]

        #putting them together in a tuple
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    #to handle the case of there not being enough pictures left in the training set to make a complete mini batch
    if m % size != 0:
        mini_batch_X = new_X[:, complete_mini_batches * size: m]
        mini_batch_Y = new_Y[:, complete_mini_batches * size: m]
        mini_bath = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_bath)

    return mini_batches


#Function to initialize the parameters for the optimization method Adam
def init_adam(params):
    L = len(params) // 2  #number of layers
    gradient_weighted_average = {}
    gradient_weighted_average_squared = {}

    #Looping through each layer
    for l in range(1, L+1):
        #Creating variables in the dictionary which are matrices of zeros of size W and b
        gradient_weighted_average["dW" + str(l)] = np.zeros(params["W" + str(l)].shape)
        gradient_weighted_average["db" + str(l)] = np.zeros(params["b" + str(l)].shape)

        #Creating variables in the dictionary which are matrices of zeros of size W and b
        gradient_weighted_average_squared["dW" + str(l)] = np.zeros(params["W" + str(l)].shape)
        gradient_weighted_average_squared["db" + str(l)] = np.zeros(params["b" + str(l)].shape)

    return gradient_weighted_average, gradient_weighted_average_squared


#Function to update the parameters after gradient descent using the learning rate and beta values to "smooth" out the curve
def update_params(params, grads, v, s, steps, lr=0.01, beta=0.9, beta2=0.99, epsilon=1e-8):
    L = len(params) // 2  #number of layers
    v_updated = {}
    s_updated = {}

    #looping through each layer
    for l in range(1, L + 1):

        #Everything below is just formulas for updating the parameters accordingly
        v["dW" + str(l)] = beta * v["dW" + str(l)] + (1 - beta) * grads['dW' + str(l)]
        v["db" + str(l)] = beta * v["db" + str(l)] + (1 - beta) * grads['db' + str(l)]

        v_updated["dW" + str(l)] = v["dW" + str(l)] / (1 - (beta)**steps)
        v_updated["db" + str(l)] = v["db" + str(l)] / (1 - (beta)**steps)

        s["dW" + str(l)] = beta2 * s["dW" + str(l)] + (1 - beta2) * (grads['dW' + str(l)])**2
        s["db" + str(l)] = beta2 * s["db" + str(l)] + (1 - beta2) * (grads['db' + str(l)])**2

        s_updated["dW" + str(l)] = s["dW" + str(l)] / (1 - (beta2)**steps)
        s_updated["db" + str(l)] = s["db" + str(l)] / (1 - (beta2)**steps)

        params["W" + str(l)] = params["W" + str(l)] - lr * (v_updated["dW" + str(l)]/(np.sqrt(s_updated["dW" + str(l)]) + epsilon))
        params["b" + str(l)] = params["b" + str(l)] - lr * (v_updated["db" + str(l)]/(np.sqrt(s_updated["db" + str(l)]) + epsilon))

    return params, v, s, v_updated, s_updated


# X is the training set, Y is the labels, layers is the number of nodes, lr is the learning rate or how fast it learns
# mini_size is the size of the mini batches(divides up the training set into sets of size mini_size)
# all of the beta values are initial decay rates and smooth the gradient
# epsilon is just a small value to avoid dividing by 0, and d_rate is the decay rate
def model_adam(X, Y, layers, lr = 0.00075, mini_size = 64, beta = 0.9, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, num_iterations = 100, d_rate = 1.0):
    t = 0  # variable for the counter for Adam
    m = X.shape[1]   # number of training examples
    params = initialize_params(layers)  # initializing the parameters
    lr0 = lr  #variable set to the initial learning rate
    v, s = init_adam(params)  #assigning variables to the initialized Adam matrices

    #Looping through each iteration
    for i in range(num_iterations):
        total_cost = 0
        minibatches = mini_batches_rdm(X, Y, mini_size)  #getting random mini batches from the function of size mini_size

        #Looping through each mini batch
        for mb in minibatches:
            (mb_X, mb_Y) = mb  #assigning values of mb_X and mb_Y from the tuple outputted
            A, caches = L_forward(mb_X, params)  #implementing forward propagation
            total_cost += cost_compute(A, mb_Y)  #computing the cost and adding it to the total cost for the current iteration
            grads = L_backward(mb_X, mb_Y, caches)  #implementing backward propagation
            t += 1  #adding one to the counter needed by Adam
            params, v, s, _, _ = update_params(params, grads, v, s, t, lr, beta1, beta2, epsilon)  #updating the parameters using adam

        average_cost = total_cost / m  #getting the average cost from all of the mini batches

        lr = (1/(1 + d_rate * np.floor(i/100))) * lr0  #formula to decay the learning rate

        if i % 1 == 0:
            print("iteration: %s, cost = %s"% (i, average_cost), lr)  #used to print the cost each iteration
            prediction(X, Y, params)

    return params


#Predicts any training set and labels inputted from it with the trained model(parameters)
def prediction(X, Y, params, keep=1):
    m = Y.shape[1]  #number of training examples

    predictions = np.zeros((1, m))  #creating an array
    A, caches = L_forward(X, params)  #using forward propagation to predict whether it is fake or real

    #Looping through each prediction and assigning a 1(fake) or 0(real) to the array called predictions
    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            predictions[0, i] = 1
        else:
            predictions[0, i] = 0

    print("accuracy was: " + str(np.mean((predictions[0, :] == Y[0, :]))))   #printing the accuracy of the model
    return predictions


if __name__ == "__main__":
    X_flattened = X.reshape(X.shape[0], -1).T #uses numpy to reshape the training set into a flattened array of size (100 * 100 * 3, 1500)
    test_X_flattened = test_X.reshape(test_X.shape[0], -1).T #uses numpy to reshape the test set into a flattened array of size (100 * 100 * 3, 100)
    print(X.shape)
    print(X_flattened.shape)

    layers = [int(X_flattened.shape[0]), 500, 400, 200, 1] #creating a variable with the number of nodes in each layer in the neural network

    # calling the function named "model_adam" which is the model with the optimization method called "Adam"
    try:
        params = model_adam(X_flattened, Y, layers, mini_size=256, lr=0.001, num_iterations=250)
    except KeyboardInterrupt:
        predictions = prediction(test_X_flattened, test_Y, params)  #calling the prediction function to predict whether something if fake or real
        sp = open("params_test", "wb")  #saving the model for later use
        pickle.dump(params, sp)

    #predictions = prediction(test_X_flattened, test_Y, params)  #calling the prediction function to predict whether something if fake or real

    
    #sp = open("params_test", "wb")  #saving the model for later use
    #pickle.dump(params, sp)

    '''
    for i in range(len(X)):
        plt.imshow(X[i])
        plt.show()
        print(predictions[0, i])
        print(Y[0, i])
    print(predictions)
    print(Y)
    '''



