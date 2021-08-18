import tensorflow as tf
from model import CNNModel
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ignore Tensorflow warning

MODEL_PATH = "model/best_acc.ckpt"
TRAIN_NUMBER = 20000  # Times of model training
choice = ""

if __name__ == '__main__':

    if tf.train.checkpoint_exists(MODEL_PATH):  # Check if there is a trained model
        choice = input('The training model already exists. Do you want to continue training or retraining? Y means '
                       'continue training, N means quit training, R means retraining').lower()
        if choice == "n":  # Quit
            exit(0)

    # Start Session
    with tf.Session() as sess:
        model = CNNModel()
        saver = tf.train.Saver()

        # Download dataset
        mnist = input_data.read_data_sets("mnist_dataset/", one_hot=True)

        # Calculation accuracy
        correct_prediction = tf.equal(tf.argmax(model.softmax, 1), tf.argmax(model.output, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Initialize Variable
        sess.run(tf.global_variables_initializer())

        # Determine whether to load the trained model
        if choice == "y":
            saver.restore(sess, MODEL_PATH)

        # Model training
        best_accuracy = 0
        print("\nNow Start!\n")
        for step in range(TRAIN_NUMBER + 1):
            batch = mnist.train.next_batch(50)
            model.train_step.run(feed_dict={model.input_shape: batch[0], model.output: batch[1], model.prob: 0.5})

            # Assessment accuracy
            if step % 10 == 0:
                eval_acc = accuracy.eval(
                    feed_dict={model.input_shape: mnist.test.images, model.output: mnist.test.labels, model.prob: 1.0})
                print("training... :{}/{} eval_acc:{:.4f}".format(step, TRAIN_NUMBER, eval_acc))

                # Save the best performance model
                if eval_acc > best_accuracy:
                    best_accuracy = eval_acc
                    saver.save(sess, MODEL_PATH)

        print("The model has been trained, and the best accuracy in the test set is:{:.4f}".format(best_accuracy))
