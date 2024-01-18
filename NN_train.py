import json

import numpy as np
import torch
import torch.nn.functional as F

train_file = 'Big_HW/train_1.json'
paras_file = 'Big_HW/params_1.json'
input_size=22
# 读取json文件
with open(train_file, 'r') as f:
    data = [json.loads(line) for line in f]
# 将1-21列数据转换为张量
x = torch.tensor([[v for k, v in d.items() if k != 'RRR'] for d in data], dtype=torch.float32)
# 将22列数据转换为张量
y = torch.tensor([d['RRR'] for d in data], dtype=torch.float32)

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.weights = []
        self.biases = []
        self.r_weights = []
        self.r_biases = []
        self.initialize_weights_and_biases()

        # Move all weights and biases to the GPU
        if torch.cuda.is_available():
            self.weights = [w.to('cuda') for w in self.weights]
            self.biases = [b.to('cuda') for b in self.biases]
            self.r_weights = [rw.to('cuda') for rw in self.r_weights]
            self.r_biases = [rb.to('cuda') for rb in self.r_biases]

    def initialize_weights_and_biases(self):
        sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        for i in range(len(sizes) - 1):
            fan_in = sizes[i]
            fan_out = sizes[i + 1]
            std = torch.sqrt(torch.tensor(2. / (fan_in + fan_out)))
            weight = torch.randn(sizes[i], sizes[i + 1]) * std
            bias = torch.zeros(sizes[i + 1])
            self.weights.append(weight)
            self.biases.append(bias)
            # 采用Adagrad算法更新学习率
            # 初始化累积梯度平方和
            self.r_weights.append(torch.zeros_like(weight))
            self.r_biases.append(torch.zeros_like(bias))

    def forward(self, x):
        Relu_Z = []
        for i in range(len(self.weights) - 1):
            x = torch.add(torch.matmul(x, self.weights[i]), self.biases[i])
            x = F.relu(x)
            Relu_Z.append(x)
        output_Z = torch.add(torch.matmul(Relu_Z[-1], self.weights[-1]), self.biases[-1])
        output = F.sigmoid(output_Z)

        return output, *Relu_Z

    def backward(self, x, y, i, output, *Relu_Z):
        # 计算损失函数关于输出的梯度
        d_loss = -(y[i] / output - (1 - y[i]) / (1 - output))

        # 计算输出关于 Z5 的梯度
        d_Z = d_loss * output * (1 - output)
        
        d_weights = []
        d_biases = []
        d_Relu_Zs = []

        for layer in reversed(range(len(Relu_Z))):
            # 计算 Z 关于 weight、bias 和 Relu_Z 的梯度
            d_weight = torch.matmul(Relu_Z[layer].view(-1, 1), d_Z.view(1, -1))
            d_bias = d_Z
            d_Relu_Z = torch.matmul(d_Z, self.weights[layer+1].t())

            d_weights.insert(0, d_weight)
            d_biases.insert(0, d_bias)
            d_Relu_Zs.insert(0, d_Relu_Z)

            # 计算 Relu_Z 关于 Z 的梯度
            d_Z = d_Relu_Z * (Relu_Z[layer] > 0).float()
        
        # 计算 Z1 关于 hiddenWeight1、hiddenBias1 和 x 的梯度
        d_weight = torch.matmul(x[i].view(-1, 1), d_Z.view(1, -1))
        d_bias = d_Z
        d_weights.insert(0, d_weight)
        d_biases.insert(0, d_bias)

        return d_weights, d_biases
    
    def loss_func(self, output, target):
        return -torch.mean(target * torch.log(output) + (1 - target) * torch.log(1 - output))

    def train(self, x, y, lr, n_iterations):
        for iteration in range(n_iterations):
            
            i = np.random.randint(0, len(x))
            x_i = x[i].to(device)  # Move input data to GPU
            y_i = y[i].to(device)  # Move target data to GPU
            
            output, *Relu_Z = self.forward(x_i)
            loss = self.loss_func(output, y_i)
            # 每迭代500次输出一次训练后output和loss
            if iteration % 2000 == 0:
                print('Iteration {}: output = {}, loss = {}'.format(iteration, output.item(), loss.item()))
            loss = self.loss_func(output, y_i)
            # Backward propagation and weights update code goes here
            d_weights, d_biases = self.backward(x, y, i, output, *Relu_Z)
            # 计算累积梯度平方和
            for i in range(len(self.weights)):
                self.r_weights[i] += d_weights[i] ** 2
                self.r_biases[i] += d_biases[i] ** 2
            
            # 更新权重和偏置
            for i in range(len(self.weights)):
                self.weights[i] -= lr / (torch.sqrt(self.r_weights[i]) + 1e-8) * d_weights[i]
                self.biases[i] -= lr / (torch.sqrt(self.r_biases[i]) + 1e-8) * d_biases[i]


    def save_params(self, filename):
        params = {
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases]
        }
        with open(filename, 'w') as f:
            json.dump(params, f)

# Usage:
device = 'cuda' if torch.cuda.is_available() else 'cpu'
nn = NeuralNetwork(input_size, hidden_sizes=[50, 50, 50, 50, 50], output_size=1)


nn.train(x, y, lr=0.001, n_iterations=100000)


nn.save_params(paras_file)
