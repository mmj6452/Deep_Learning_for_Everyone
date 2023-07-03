import tensorflow as tf

node1 = tf.constant(3.0,tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1,node2)

print("node1 = " , node1, "node2 = ", node2)
print("node3 = ", node3)

@tf.function
def adder(a,b):
    return a+b

a = tf.constant([1,3])
b = tf.constant([2,4])

print(adder(a,b))