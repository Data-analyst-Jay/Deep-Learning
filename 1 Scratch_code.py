import numpy as np
x = np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])
y = np.array([[1],[1],[0]])

def signmoid(x):
    return 1/(1 + np.exp(-x))

def der_sigmoid(x):
    return x*(1-x)

epoch = 10000
lr = 0.2
inputlayer_neurons = x.shape[1]
hiddenlayer_neurons = 3
output_neurons = 1

# initialising weight and bias
wh = np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh = np.random.uniform(size=(1,hiddenlayer_neurons))
wout = np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout = np.random.uniform(size=(1,output_neurons))

#Training the model
for i in range(epoch):
    
    #Forward propagation
    hidden_layer_input1 = np.dot(x,wh)
    hidden_layer_input = hidden_layer_input1 + bh
    hidden_layer_activations = signmoid(hidden_layer_input)
    output_layer_input1 = np.dot(hidden_layer_activations,wout)
    output_layer_input = output_layer_input1 + bout
    output = signmoid(output_layer_input)

    #Backward propagation
    E = y-output
    slope_output_layer = der_sigmoid(output)
    slope_hidden_layer = der_sigmoid(hidden_layer_activations)
    d_output = E * slope_output_layer
    error_at_hiddenlayer = d_output.dot(wout.T)
    d_hiddenlayer = error_at_hiddenlayer * slope_hidden_layer
    wout += hidden_layer_activations.T.dot(d_output) * lr
    bout += np.sum(d_output, axis=0, keepdims=True) * lr
    wh += x.T.dot(d_hiddenlayer) * lr
    bh += np.sum(d_hiddenlayer, axis= 0 , keepdims=True) * lr

print(output)