import tensorflow as tf


# Use Convolutional Neural Networks
class CNNModel:

    # Initialization
    def __init__(self):
        # Placeholder
        self.input_shape = tf.placeholder(tf.float32, shape=[None, 784])  # 28x28=784
        self.output = tf.placeholder(tf.float32, shape=[None, 10])  # classification 0-9

        # Adjust the shape of the input tensor and adjust it to a 28*28 picture
        input_shape_all = tf.reshape(self.input_shape, [-1, 28, 28, 1])

        # Convolution layer 1
        # Define the first layer of filter and bias, the dimension of the bias is 32
        conv1_weight = self.init_weight([5, 5, 1, 32])
        conv1_bias = self.init_bias([32])

        # The first level convolution operation is carried out to get the result of linear variation, and then the
        # nonlinear mapping is carried out by using the relu rule to get the first level convolution result
        conv1_relu = tf.nn.relu(self.creat_conv2d(input_shape_all, conv1_weight) + conv1_bias)

        # The pooling results are obtained by the maximum pooling method
        conv1_pool = self.max_pool(conv1_relu)  # 28x28 -> 14x14

        # Convolution layer 2, Basically the same as the first layer
        conv2_weight = self.init_weight([5, 5, 32, 64])
        conv2_bias = self.init_bias([64])
        conv2_relu = tf.nn.relu(self.creat_conv2d(conv1_pool, conv2_weight) + conv2_bias)
        conv2_pool = self.max_pool(conv2_relu)  # 14x14 -> 7x7

        # Fully-Connected layer
        # Define the weight and bias of the fully connected layer
        fc1_weight = self.init_weight([7 * 7 * 64, 1024])
        fc1_bias = self.init_bias([1024])

        # Adjust the pooled data of the second layer to a 7 × 7 × 64 Vector
        fc1_pool = tf.reshape(conv2_pool, [-1, 7 * 7 * 64])

        # Multiply the weight of the fully connected layer by matrix, and then perform nonlinear mapping to obtain a
        # 1024-dimensional vector
        fc1_relu = tf.nn.relu(tf.matmul(fc1_pool, fc1_weight) + fc1_bias)

        # Random inactivation layer
        self.prob = tf.placeholder("float")
        fc1_dropout = tf.nn.dropout(fc1_relu, self.prob)

        # Softmax Output layer
        # Weight and bias of the output layer
        fc2_weight = self.init_weight([1024, 10])
        fc2_bias = self.init_bias([10])
        self.softmax = tf.nn.softmax(tf.matmul(fc1_dropout, fc2_weight) + fc2_bias)  # Get the output

        # Calculate loss
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.output, logits=self.softmax))

        # Gradient descent
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # Initialize Weight
    def init_weight(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # Initialize bias
    def init_bias(self, shape):
        initial = tf.constant(0, 1, shape=shape)
        return tf.Variable(initial)

    # Create Convolution layer
    def creat_conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # Maximum pooling layer
    def max_pool(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
