import tensorflow as tf

classes = 4
labels = tf.constant([[0,1],[1,0]])
output = tf.one_hot(labels,classes, on_value = 2, off_value = 3, axis = 0)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    output = sess.run(output)
    print("output of one-hot is : " , output)
