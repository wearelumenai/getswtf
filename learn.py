import tensorflow as tf
from tensorflow.python.data import Iterator

HEIGHT = 32
WIDTH = 32
CHANNELS = 3

IN_TRAIN = 'data/train.in'
OUT_TRAIN = 'data/train.out'
IN_TEST = 'data/test.in'
OUT_TEST = 'data/test.out'


def mk_weight_variable(shape):
    """
    Build and initialize a variable node that represents a weight matrix.

    :param shape: the matrix shape
    :return: the weight matrix variable
    """
    initial = tf.truncated_normal(shape, stddev=.1)
    return tf.Variable(initial)


def mk_bias_variable(shape):
    """
    Build and initialize a variable node that represents a bias vector.

    :param shape: the vector shape
    :return: the bias vector variable
    """
    initial = tf.truncated_normal(shape=shape, stddev=.1)
    return tf.Variable(initial)


def mk_parse(in_raw, out_raw):
    """
    Build a sub-graph that parses images retrieved from persistent storage.

    Pixel are encoded from 0 to 255 (unsigned 8 bit int) and the image is properly resized.

    :param in_raw: the raw input image tensor
    :param out_raw: the raw input image tensor
    :return: - a single channel image tensor for input
             - a single channel image tensor for output
    """
    # encoding
    in_decoded = tf.decode_raw(in_raw, out_type=tf.uint8)
    out_decoded = tf.decode_raw(out_raw, out_type=tf.uint8)

    # casting
    in_cast = tf.subtract(tf.to_float(in_decoded), tf.constant(127.0, dtype=tf.float32))
    out_cast = tf.to_int64(out_decoded)

    # reshaping
    in_image = tf.reshape(in_cast, [HEIGHT, WIDTH, CHANNELS])

    return in_image, out_cast


def mk_batch(epochs=4, batch_size=50, buf_size=7000):
    """
    Build a sub graph that represents the input and output data
    and iterators to feed the graph either with train or test data.

    Four objects are returned :
     - Two tensors : one for input images and another for output images.
     - Two iterators : one for training set and another for test set.

    The tensors will be used to create the graph. The iterators will be used to
    feed the tensors with the train or test data.

    The iterators may be initialized with either training or test data.

    :param epochs the number of epochs
    :param batch_size the size of the mini-batches
    :param buf_size the size of the shuffling buffer
    :return: the batch iterators and initializers
    """

    # Training dataset, collect input and output from corresponding files and zip them together
    train_in = tf.data.FixedLengthRecordDataset(IN_TRAIN, WIDTH * HEIGHT * CHANNELS)
    train_out = tf.data.FixedLengthRecordDataset(OUT_TRAIN, 1)
    train_data = tf.data.Dataset.zip((train_in, train_out)) \
        .map(mk_parse) \
        .shuffle(buf_size) \
        .batch(batch_size) \
        .repeat(epochs)

    # Test dataset, collect input and output from corresponding files and zip them together
    test_in = tf.data.FixedLengthRecordDataset(IN_TEST, WIDTH * HEIGHT * CHANNELS)
    test_out = tf.data.FixedLengthRecordDataset(OUT_TEST, 1)
    test_data = tf.data.Dataset.zip((test_in, test_out)) \
        .map(mk_parse) \
        .shuffle(buf_size) \
        .batch(batch_size)

    # Build a feedable iterator that obey to the above datasets structure
    handle = tf.placeholder(tf.string, shape=[])
    iterator = Iterator.from_string_handle(handle, train_data.output_types, train_data.output_shapes)
    in_batch, out_batch = iterator.get_next()

    train = train_data.make_one_shot_iterator()
    test = test_data.make_initializable_iterator()

    # The size of the mini-batch
    size = tf.shape(in_batch)[0]

    return in_batch, out_batch, size, train, test, handle


def mk_hidden_layer(shape, in_data):
    """
    Build a sub-graph that represents a hidden layer.

    The hidden layer consists of a batch normalization, a convolution and a RELU.
    During the training phase, the batch normalization is computed from the current input moments also used to
    update moving averages. When it comes to prediction or validation, the batch normalization uses the moving averages.

    :param shape: the shape of the convolution filters : ``[filter_height, filter_width, in_channels, out_channels]``
    :param in_data: a tensor that represents the mini-batches input
    :return: a tensor that represents the output of the hidden layer
    """

    # build weights and biases nodes
    w = mk_weight_variable(shape)
    b = mk_bias_variable(shape[-1:])

    # convolution and activation
    conv = tf.nn.conv2d(in_data, w, strides=[1, 1, 1, 1], padding='SAME')
    model = tf.add(conv, b)
    hidden = tf.nn.relu(model)

    return hidden


def mk_linear_layer(shape, in_data):
    """
    Build a sub-graph that represents the output layer.

    The output layer consists of fully connected neurons

    :param shape: the shape of the fully connected layer : ``[size_in, size_out]``
    :param in_data: a tensor that represents the mini-batches input
    :return: a tensor that represents the output
    """
    w = mk_weight_variable(shape)
    b = mk_bias_variable(shape[-1:])

    reshaped = tf.reshape(in_data, [-1, shape[0]])
    return tf.add(tf.matmul(reshaped, w), b)


def mk_eval(in_data, out_data):
    """
    Build the evaluation metrics sub-graphs.

    :param in_data:
    :param out_data:
    :return: tensors and operations that represents :
     - the loss
     - the accuracy
     - the optimisation operation
     - the optimisation step
     - the learning rate
    """
    one_hot = tf.one_hot(out_data, 2)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot, logits=in_data)
    loss = tf.reduce_mean(cross_entropy)

    is_correct = tf.equal(out_data, tf.argmax(in_data, axis=1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    return loss, accuracy


def mk_learning(loss):
    """
    Build the learning sub-graph.

    :param in_data:
    :param out_data:
    :return: tensors and operations that represents :
     - the optimisation operation
     - the optimisation step
     - the learning rate
    """
    learning_rate = tf.placeholder(tf.float32, shape=[], name="l_rate")
    step = tf.Variable(0, trainable=False)
    optim = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss, global_step=step)
    return learning_rate, optim, step


def mk_model(in_data, out_data):
    """
    Build the model.
    """
    hidden1 = mk_hidden_layer([3, 3, CHANNELS, 4], in_data)
    model = mk_linear_layer([WIDTH * HEIGHT * 4, 2], hidden1)
    loss, accuracy = mk_eval(model, out_data)
    learning_rate, optim, step = mk_learning(loss)
    return model, loss, accuracy, optim, step, learning_rate


if __name__ == '__main__':

    print('Building model...\n')
    # bind data to model
    in_data, out_data, size, train_iter, test_iter, handle = mk_batch()
    model, loss, accuracy, optim, step, learning_rate = mk_model(in_data, out_data)

    # start learning
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('\nStart learning...')
        train_handle = sess.run(train_iter.string_handle())
        test_handle = sess.run(test_iter.string_handle())


        def do_test():
            """
            Test the model with the test set
            :return: the accuracy
            """
            test_feed = {handle: test_handle}
            test_accuracy = 0
            test_size = 0

            sess.run(test_iter.initializer)

            while True:
                try:
                    accuracy_val, size_val = sess.run([accuracy, size], feed_dict=test_feed)
                    test_accuracy = (test_accuracy * test_size + accuracy_val * size_val) / \
                                    (test_size + size_val)
                    test_size = test_size + size_val
                except tf.errors.OutOfRangeError:
                    # test terminated
                    break

            return test_accuracy


        def do_train():
            """
            Train the model with the train set
            """
            train_feed = {handle: train_handle, learning_rate: .1}

            while True:
                try:
                    step_val = sess.run(step)

                    if (step_val + 1) % 30 != 0:
                        sess.run(optim, feed_dict=train_feed)
                    else:
                        _, train_accuracy = sess.run([optim, accuracy], feed_dict=train_feed)
                        test_accuracy = do_test()

                        print('train accuracy: {}, test accuracy: {}'.format(train_accuracy, test_accuracy))

                except tf.errors.OutOfRangeError:
                    # training terminated
                    break


        do_train()
