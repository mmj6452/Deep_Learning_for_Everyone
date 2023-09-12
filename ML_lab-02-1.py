import tensorflow as tf

#학습데이터
x_train = [1, 2, 3]
y_train = [1, 2, 3]

#W와 b를 각각 무작위 값으로 초기화
W = tf.Variable(tf.random.normal([1]), name = 'weight')
b = tf.Variable(tf.random.normal([1]), name = 'bias')

def cost_fn():
    #가설
    hypothesis = x_train * W + b
    #비용함수
    coat = tf.reduce_mean(tf.square(hypothesis - y_train))
    return coat
#경사하강법의 정의
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 0.01)

for step in range(2001):
    #비용함수를 최소화하는 W와 b를 찾아내는 과정
    optimizer.minimize(cost_fn, var_list = [W, b],feed_dict = {x_train:[1,2,3], y_train:[1,2,3]})
    if step % 20 == 0:
        cost_val = cost_fn()
        print(step, cost_val.numpy(), W.numpy(), b.numpy())