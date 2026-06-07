import pickle
import numpy as np
import matplotlib.pylab as plt
import sys,os
from PIL import Image
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # 親ディレクトリのファイルをインポートするための設定
from dataset.mnist import load_mnist


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5]) # 重みとバイアスだけがANDと違う！
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5]) # 重みとバイアスだけがANDと違う！
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

def step_function(x):
    y = x > 0   
    return y.astype(int) 
    #astype()メソッドでは、引数に希望する型を指定
    #xがNumPyの配列でもいける

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0,x)

def init_network():
    network = {}
    network["W1"] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network["b1"] = np.array([0.1,0.2,0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def identity_function(x):
    return x

def forward(network,x):
    W1,W2,W3 = network["W1"],network["W2"],network["W3"]
    b1,b2,b3 = network["b1"],network["b2"],network["b3"]

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    return y

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) #オーバーフロー対策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

def get_data():
    (x_train,t_train),(x_test,t_test) = load_mnist(normalize=True,flatten=True,one_hot_label=False)
    #load_mnist関数は、「(訓練画像, 訓練ラベル), (テスト画像, テストラベル)」という形式で、
    #読み込んだMNISTデータを返します。
    return x_test,t_test


def init_network():
    pkl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_weight.pkl")
    with open(pkl_path,"rb") as f:
        network = pickle.load(f)

    #init_network()では、pickleファイルのsample_weight.pklに
    #保存された学習済みの重みパラメータを読み込みます。
    return network

def predict(network,x):
    W1,W2,W3 = network["W1"],network["W2"],network["W3"]
    b1,b2,b3 = network["b1"],network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

def sum_squared_error(y,t): #2乗和誤差
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y,t): #交差エントロピー誤差
    if y.ndim == 1: # ndim は　nの次元数
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

def numerical_diff(f,x): #数値微分
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

def numerical_gradient(f,x): #勾配
    h = 1e-4
    grad = np.zeros_like(x) #xと同じ形状の中身がすべて０の配列を生成

    for idx in range(x.size):
        tmp_val = x[idx]
        #f(x+h)の計算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        #f(x-h)の計算
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val #値を元に戻す

    return grad

def gradient_descent(f,init_x,lr=0.01,step_num=100): #勾配降下法
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f,x)
        x -= lr * grad

    return x


def function_2(x):
    return x[0] ** 2 + x[1]**2

class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        #重みの初期化
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size,hidden_size)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y
    
    # x:入力データ, t:教師データ
    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

# x:入力データ, t:教師データ
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads
    
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self,x,y):
        self.x = x
        self.y = y
        out = x * y

        return out
    
    def backward(self,dout): #dout　上流から伝わってきた微分
        dx = dout * self.y
        dy = dout * self.x

        return dx,dy

class AddLayer:
    def __init__(self):
        pass

    def forward(self,x,y):
        out = x + y
        return out
    
    def backward(self,dout):
        dx = dout * 1
        dy = dout * 1
        return dx,dy
    
class Relu:
    def __init__(self):
        self.mask = None  #mask

    def forward(self,x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out
    
    def bavckward(self,dout):
        dout[self.mask] = 0
        dx = dout

        return dx
    
x = np.array([[1.0,-0.5],[-2.0,3.0]])


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self,x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out
    
    def backward(self,dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx