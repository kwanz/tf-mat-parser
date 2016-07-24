import tensorflow as tf
import numpy as np
import scipy.io

test_mat = {}
test_matrix = np.random.rand(2, 3, 4)
test_mat['test'] = test_matrix
scipy.io.savemat('test.mat', test_mat)
print('Generated matrix:')
print(test_matrix)

parse_mat_module = tf.load_op_library('parse_mat.so')
test_parse_tensor = parse_mat_module.parse_mat('test.mat', 'test', dtype=tf.float64)
sess = tf.InteractiveSession()
test_parse = sess.run(test_parse_tensor)
print('Parsed matrix:')
print(test_parse)
