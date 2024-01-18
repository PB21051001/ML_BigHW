import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
paras_file = 'Big_HW/params_1.json'
test_file = 'Big_HW/test_1.json'

class TestModel:
    def __init__(self, weights, biases):
        self.weights = [torch.tensor(w, dtype=torch.float32) for w in weights]
        self.biases = [torch.tensor(b, dtype=torch.float32) for b in biases]

    def forward(self, x):
        for i in range(len(self.weights) - 1):
            x = F.relu(torch.add(torch.matmul(x, self.weights[i]), self.biases[i]))
        return torch.sigmoid(torch.add(torch.matmul(x, self.weights[-1]), self.biases[-1]))




# 读取json文件
with open(test_file, 'r') as f:
    data = [json.loads(line) for line in f]

# 将1-21列数据转换为张量
test_x = torch.tensor([[v for k, v in d.items() if k != 'RRR'] for d in data], dtype=torch.float32)
# 将22列数据转换为张量
test_y = torch.tensor([d['RRR'] for d in data], dtype=torch.float32)

#读入参数
with open(paras_file, 'r') as f:
    params = json.load(f)

model = TestModel(params['weights'], params['biases'])

y_true = []
y_pred = []
# 对测试集进行预测
for i in range(len(test_x)):
    output = model.forward(test_x[i])
    y_pred.append(output.item())
    y_true.append(test_y[i].item())

# 计算准确率，精确率，召回率，F1值
accuracy = accuracy_score(y_true, np.round(y_pred))
precision = precision_score(y_true, np.round(y_pred))
recall = recall_score(y_true, np.round(y_pred))
f1 = f1_score(y_true, np.round(y_pred))
print("accuracy: ", accuracy)
print("precision: ", precision)
print("recall: ", recall)
print("f1: ", f1)

