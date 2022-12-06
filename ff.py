#%%
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
from tqdm import tqdm
#%%

class Sigmoid:
    def __call__(self, x):
        return (1 / (1 + np.exp(-x)))

    def gradient(self, x):
        return self(x) * (1 - self(x))

sigmoid = Sigmoid()


class ReLU:
    def __call__(self, x):
        return np.maximum(0, x)
    
    def gradient(self,x):
        return np.maximum(0, x)

    
class Linear:
    def __call__(x):
        return x
    
    def gradient(x):
        return x


class DenseLayer:
    def __init__(self, fan_in, fan_out, activation=Linear()):
        self.w = np.random.normal(size=(fan_in, fan_out)) * 0.001
        self.activation = activation

    def forward(self, x):
        '''
        forward pass
        '''
        activation = self.activation(x @ self.w)
        return activation

    def normalize(self, x):
        '''
        normalize forward pass for next layer
        '''
        example_norms = np.linalg.norm(x, axis=1, keepdims=True)    
        return x / example_norms

    def gradient(self, x, y):
        '''
        forward pass is y = relu(wx)
        the network aims to minimize its activity for
        positive data, so:
        a = wx
        y = relu(a)
        e = 0.5 (y @ y) 
        goodness = sigmoid(e)

        dg/de = sigmoid'(e)
        de/dy = y
        dy/da = relu'(a)
        da/dw = x

        -->

        dg/dw = dg/de y dy/da x

        y - [batchsize fan out] 
        x - [batchsize fan in]
        w = [fan_in fan_out]
        '''
        
        e = 0.5 * np.sum(y * y, axis=1, keepdims=True)  # [batchsize, 1]
        dgde = sigmoid.gradient(e)                      # [batchsize, 1]
        dyda = self.activation.gradient(x @ self.w)     # [batchsize, fan_out]
        grads = x.T @ (dgde * y * dyda)
        return grads

    def compute_update(self, pos_data=None, neg_data=None):
        if pos_data is None and neg_data is None:
            raise ValueError('must supply at least one type of data')
        # compute update for positive data
        if not pos_data is None:
            ypos = self.forward(pos_data)
            pos_grads = self.gradient(pos_data, ypos)
            pos_grads /= len(pos_data)
        else:
            pos_grads = np.zeros(shape=self.w.shape)
        if not neg_data is None:
            yneg = self.forward(neg_data)
            neg_grads = self.gradient(neg_data, yneg)
            neg_grads *= -1
            neg_grads /= len(neg_data)
        else:
            neg_grads = np.zeros(shape=self.w.shape)

        grads = pos_grads + neg_grads 
        return grads


class SGD:
    def __init__(self, variables, lr):
        self.variables = variables
        self.lr = lr
    
    def apply(self, grads):
        for v,g in zip(self.variables, grads):
            v -= self.lr * g


class Network:
    def __init__(self, layers) -> None:
        self.layers = layers
    
    def __call__(self, x):
        activity = []
        for l in self.layers:
            y = l.forward(x)
            activity.append(y)
            ynorm = l.normalize(y)
            x = ynorm  # normalized output is input for next layer
        return activity
    
    def compute_update(self, pos_data=None, neg_data=None):
        if pos_data is None and neg_data is None:
            raise ValueError("must supply at least one type of data.")
        grads = []
        for l in self.layers:
            grad = l.compute_update(pos_data, neg_data)
            if pos_data is not None:
                pos_data = l.normalize(l.forward(pos_data))
            if neg_data is not None:
                neg_data = l.normalize(l.forward(neg_data))
            grads.append(grad)
        return grads

    def energy(self, x):
        activities = self(x)
        energy = 0
        for a in activities:
            energy += np.linalg.norm(a, axis=1)
        return energy
    
    def goodness(self, x):
        energies = []
        for a in self(x):
            energies.append(np.linalg.norm(a))
        goodnesses = [np.mean(sigmoid(e)) for e in energies]
        return goodnesses


#%%
# testing: distinguish 4 from 2
(trainX, trainy), (testX, testy) = mnist.load_data()
trainX = np.reshape(trainX, (len(trainX), -1)) / 255.
testX = np.reshape(testX, (len(testX), -1)) / 255.
twos = trainX[trainy == 2]  # negative examples
fours = trainX[trainy == 4]  # positive examples

two_test = testX[testy == 2]
four_test = testX[testy == 4]

layer1 = DenseLayer(784, 512, activation=ReLU())
layer2 = DenseLayer(512,256, activation=ReLU())
layers = [layer1, layer2]
net = Network(layers)
optim = SGD([l.w for l in net.layers], lr=0.01)

print('Init.')
energy_four = np.mean(net.energy(four_test))
energy_two = np.mean(net.energy(two_test))
print(f"Energy pos. data: {energy_four}. Energy neg. data: {energy_two}.")

pos_energies = []
neg_energies = []

pos_goodnesses = []
neg_goodnesses = []

def batch_data(X, batchsize):
    nsamples = len(X)
    split_indices = np.arange(start=batchsize, stop=nsamples, step=batchsize)
    Xbatched = np.split(X, split_indices)
    return Xbatched

def train_data_iterator(pos_data, neg_data, batchsize):
    np.random.shuffle(pos_data)
    np.random.shuffle(neg_data)
    pos_batched = batch_data(pos_data, batchsize)
    neg_batched = batch_data(neg_data, batchsize)
    batched = [(x, 1) for x in pos_batched] + [(x, 0) for x in neg_batched]
    np.random.shuffle(batched)
    for b in batched:
        yield b

for i in range(80):
    print(f"Epoch: {i}")
    for d in train_data_iterator(fours, twos, 128):
        x, l = d
        if l == 0:
            grads = net.compute_update(neg_data=x)
        else:
            grads = net.compute_update(pos_data=x)
        optim.apply(grads)
    energy_four = np.mean(net.energy(four_test))
    energy_two = np.mean(net.energy(two_test))
    pos_energies.append(energy_four)
    neg_energies.append(energy_two)
    pos_goodness = net.goodness(four_test)
    neg_goodness = net.goodness(two_test)
    pos_goodnesses.append(np.mean(pos_goodness))
    neg_goodnesses.append(np.mean(neg_goodness))
    print(f"Energy pos. data: {energy_four}. Energy neg. data: {energy_two}.")
# %%

plt.figure()
plt.plot(pos_energies, label='positive data')
plt.plot(neg_energies, label='negative data')
plt.legend()
plt.title("Energies over training")

plt.figure()
plt.plot(pos_goodnesses, label='goodness positive data')
plt.plot(neg_goodnesses, label='goodness negative data')
plt.title('goodnesses')
plt.legend()
plt.show()

# %%
