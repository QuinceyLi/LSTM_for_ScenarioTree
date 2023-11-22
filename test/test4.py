
##### tf计算gamma函数梯度######
import tensorflow as tf

print("version:",tf.__version__)

x = tf.constant(5, dtype=tf.float64)

y = tf.math.igamma(x, 1e4)

y2 = tf.math.lgamma(x)

print(y2)

with tf.GradientTape() as tape:
    x = tf.constant(5, dtype=tf.float64)
    
    y2 = tf.math.exp(tf.math.lgamma(x))

grads = tape.gradient(y2, x) # None

print("tf grad:",grads)

#######pytorch计算######
import torch
x = torch.tensor(5, dtype=torch.float64, requires_grad=True)
o = torch.tensor(1e4, dtype=torch.float64, requires_grad=True)

y = torch.igamma(input=x,other=o)

z = torch.exp(x)

print(y)

# y.backward()
z.backward()

# print("dy/dx:", x.grad)
print("dz/do", x.grad)