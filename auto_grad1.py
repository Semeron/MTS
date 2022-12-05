

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import torch
import numpy as np


# 定义一个继承了Function类的子类，实现y=f(x)的正向运算以及反向求导
class sqrt_and_inverse(torch.autograd.Function):
    '''
    forward和backward可以定义成静态方法，向定义中那样，也可以定义成实例方法
    '''

    # 前向运算
    @staticmethod
    def forward(ctx, input_x, input_y):
        '''
        self.save_for_backward(input_x,input_y) ,这个函数是定义在Function的父类_ContextMethodMixin中
             它是将函数的输入参数保存起来以便后面在求导时候再使用，起前向反向传播中协调作用
        '''
        ctx.save_for_backward(input_x, input_y)
        output = torch.sqrt(input_x) + torch.reciprocal(input_x) + 2 * torch.pow(input_y, 2)
        return output

    @staticmethod
    def backward(ctx, grad_output):   #grad_output就是backward（gradient）的第一个参数gradient
        input_x, input_y = ctx.saved_tensors  # 获取前面保存的参数,也可以使用self.saved_variables
        grad_x = grad_output * (torch.reciprocal(2 * torch.sqrt(input_x)) - torch.reciprocal(torch.pow(input_x, 2)))
        grad_y = grad_output * (4 * input_y)

        return grad_x, grad_y  # 需要注意的是，反向传播得到的结果需要与输入的参数相匹配


# 由于sqrt_and_inverse是一个类，我们为了让它看起来更像是一个pytorch函数，需要包装一下
def sqrt_and_inverse_func(input_x, input_y):
    return sqrt_and_inverse.apply(input_x, input_y)  # 这里是对象调用的含义，因为function中实现了__call__


x = torch.tensor(3.0, requires_grad=True)  # 标量
y = torch.tensor(2.0, requires_grad=True)

print('开始前向传播')
z = sqrt_and_inverse_func(x, y)

print('开始反向传播')
gradient=torch.tensor(2)
z.backward(gradient)


print(x.grad)
print(y.grad)
'''运行结果为：
开始前向传播
开始反向传播
tensor(0.1776)
tensor(8.)
'''

