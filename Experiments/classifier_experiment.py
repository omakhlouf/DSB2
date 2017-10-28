import tensorflow as tf
import numpy as np
from nnutils import *

import pandas, os

def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def make_class_label(y):
    ''' Return an encoded version of y in one of 10 percentile classes '''
    y = np.array(y).astype(int)
    encodings = np.zeros((y.shape[0], 2))
    for i in range(y.shape[0]):
        if y[i] < 100.0:
            encodings[i, 0] = 1.
        # elif y[i] < 38.46:
        #     encodings[i, 1] = 1.
        # elif y[i] < 44.39:
        #     encodings[i, 2] = 1.
        # elif y[i] < 48.78:
        #     encodings[i, 3] = 1.
        # elif y[i] < 52.5:
        #     encodings[i, 4] = 1.
        # elif y[i] < 57.95:
        #     encodings[i, 5] = 1.
        # elif y[i] < 65.07:
        #     encodings[i, 6] = 1.
        # elif y[i] < 72.3:
        #     encodings[i, 7] = 1.
        # elif y[i] < 77.3:
        #     encodings[i, 8] = 1.
        # elif y[i] < 85.54:
        #     encodings[i, 9] = 1.
        # elif y[i] < 92.86:
        #     encodings[i, 10] = 1.
        # elif y[i] < 103.952:
        #     encodings[i, 11] = 1.
        # elif y[i] < 114.384:
        #     encodings[i, 12] = 1.
        # elif y[i] < 121.92:
        #     encodings[i, 13] = 1.
        # elif y[i] < 129.68:
        #     encodings[i, 14] = 1.
        # elif y[i] < 137.94:
        #     encodings[i, 15] = 1.
        # elif y[i] < 149.732:
        #     encodings[i, 16] = 1.
        # elif y[i] < 158.03:
        #     encodings[i, 17] = 1.
        # elif y[i] < 169.9:
        #     encodings[i, 18] = 1.
        # elif y[i] < 178.14:
        #     encodings[i, 19] = 1.
        # elif y[i] < 190.12:
        #     encodings[i, 20] = 1.
        # elif y[i] < 202.824:
        #     encodings[i, 21] = 1.
        # elif y[i] < 220.748:
        #     encodings[i, 22] = 1.
        # elif y[i] < 249.016:
        #     encodings[i, 23] = 1.
        else:
            encodings[i, 1] = 1.

    return encodings

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) #Random noise for symmetry breaking
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') #Stride of 1

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #2x2 max pooling, stride = 2


from sacred import Experiment
# from sacred.observers import MongoObserver
ex = Experiment('DSB CONVNET EXPERIMENT')
# ex.observers.append(MongoObserver.create())

@ex.config
def config():
    RUN_NAME = 'CRPS-MODEL-3.0'
    DATA_DIR = 'netdata'
    ITERS = 100000
    START_ITER = 0
    MODEL_LOAD_PATH = None
    PIC_WIDTH = 32
    ### Architectural Hyperparameters
    DEPTH_1 = 8         # The output depth of the first convolutional layer
    DEPTH_2 = 16         # The output depth of the second convolutional layer
    DEPTH_3 = 20         # The output depth of the second convolutional layer
    DEPTH_4 = 24        # The output depth of the second convolutional layer
    DEPTH_5 = 30        # The output depth of the second convolutional layer
    DEPTH_6 = 40        # The output depth of the second convolutional layer

    NUM_OUTPUTS = 10    # Number of output classes in the softmax layer
    KERNEL_X = 3         # The width of the convolution kernel (using same for 1st and 2nd layers)
    KERNEL_Y = 3         # The height of the convolution kernel (using same for 1st and 2nd layers)
    mu = 0.0001
    LEARNING_RATE = 1e-4

    REGULARIZE_BIAS = False

    NUM_INPUTS = 9  
    NUM_REPS = 144

    TRAIN_LABEL_NOISE_STD = 2.
    TRAIN_LABEL_SMOOTHING_STD = 0.
    DATA_AUGMENTATION = True
    DIASTOLE = False

@ex.named_config
def classifier():
    NUM_INPUTS = 9       # Number of input channels
    NUM_REPS = 144
    TRAIN_LABEL_NOISE_STD = 0.0001
    TRAIN_LABEL_SMOOTHING_STD = 0.
    LEARNING_RATE = 2e-3
    mu = 0.1
    DATA_AUGMENTATION = True
    RUN_NAME = 'CLASSIFIER-EXP'

@ex.named_config
def two_classifier():
    NUM_INPUTS = 9       # Number of input channels
    NUM_OUTPUTS = 2
    NUM_REPS = 144
    TRAIN_LABEL_NOISE_STD = 0.0001
    TRAIN_LABEL_SMOOTHING_STD = 0.
    LEARNING_RATE = 1e-3
    mu = 0.02
    DATA_AUGMENTATION = True
    RUN_NAME = 'TWO-CLASS-EXP3'
    DEPTH_1 = 4         # The output depth of the first convolutional layer
    DEPTH_2 = 4         # The output depth of the second convolutional layer
    DEPTH_3 = 4         # The output depth of the second convolutional layer
    DEPTH_4 = 2        # The output depth of the second convolutional layer
    DEPTH_5 = 2        # The output depth of the second convolutional layer
    DEPTH_6 = 2        # The output depth of the second convolutional layer

@ex.named_config
def twenty_five_classifier():
    NUM_INPUTS = 9       # Number of input channels
    NUM_OUTPUTS = 25
    NUM_REPS = 144
    TRAIN_LABEL_NOISE_STD = 0.0001
    TRAIN_LABEL_SMOOTHING_STD = 0.
    LEARNING_RATE = 1e-3
    mu = 0.05
    DATA_AUGMENTATION = True
    RUN_NAME = 'CLASS-25-EXP'
    DEPTH_1 = 8         # The output depth of the first convolutional layer
    DEPTH_2 = 16         # The output depth of the second convolutional layer
    DEPTH_3 = 20         # The output depth of the second convolutional layer
    DEPTH_4 = 24        # The output depth of the second convolutional layer
    DEPTH_5 = 30        # The output depth of the second convolutional layer
    DEPTH_6 = 40       # The output depth of the second convolutional layer

@ex.capture
def load_data(DATA_DIR):
    train = np.load(os.path.join(DATA_DIR, 'standardized_train.npy'))
    train_labels = np.load(os.path.join(DATA_DIR, 'labels_train.npy'))
    test = np.load(os.path.join(DATA_DIR, 'standardized_test.npy'))
    test_labels = np.load(os.path.join(DATA_DIR, 'labels_test.npy'))


    return train, train_labels, test, test_labels

@ex.capture
def construct_graph(PIC_WIDTH,DEPTH_1,DEPTH_2,DEPTH_3,DEPTH_4,DEPTH_5,DEPTH_6,
                    NUM_INPUTS,NUM_OUTPUTS,KERNEL_X,KERNEL_Y,mu,
                    LEARNING_RATE,NUM_REPS,REGULARIZE_BIAS):
    graph = tf.Graph()
    with graph.as_default():
        ####################################### INPUT/OUTPUT PLACEHOLDERS ##############################################
        x = tf.placeholder(tf.float32, shape=[None, PIC_WIDTH, PIC_WIDTH, NUM_INPUTS]) #Placeholder for the input images
        y_ = tf.placeholder(tf.float32, shape=[None, NUM_OUTPUTS]) #Placeholder for the label cdfs

        ####################################### FIRST CONVOLUTIONAL LAYER ##############################################
        # The weight tensor has dimensions [kernel_size_x, kernel_size_y, num_input_channels, num_output_channels]
        W_conv1 = weight_variable([KERNEL_X, KERNEL_Y, NUM_INPUTS, DEPTH_1])
        b_conv1 = bias_variable([DEPTH_1])
        #Take the input image, reshape it to a 4D tensor with dimensions: [_, image_width, image_height, num_channels]
        #x_image = tf.reshape(x, [-1,32,32,3])
        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        ####################################### SECOND CONVOLUTIONAL LAYER ##############################################
        W_conv2 = weight_variable([KERNEL_X, KERNEL_Y, DEPTH_1, DEPTH_2])
        b_conv2 = bias_variable([DEPTH_2])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        ####################################### THIRD CONVOLUTIONAL LAYER ##############################################
        W_conv3 = weight_variable([KERNEL_X, KERNEL_Y, DEPTH_2, DEPTH_3])
        b_conv3 = bias_variable([DEPTH_3])

        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = max_pool_2x2(h_conv3)

        ####################################### FOURTH CONVOLUTIONAL LAYER ##############################################
        W_conv4 = weight_variable([KERNEL_X, KERNEL_Y, DEPTH_3, DEPTH_4])
        b_conv4 = bias_variable([DEPTH_4])

        h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
        h_pool4 = max_pool_2x2(h_conv4)
        # ####################################### FIFTH CONVOLUTIONAL LAYER ##############################################
        #This layer uses a 1x1 convolution
        W_conv5 = weight_variable([1, 1, DEPTH_4, DEPTH_5])
        b_conv5 = bias_variable([DEPTH_5])

        h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
        h_pool5 = tf.nn.avg_pool(h_conv5,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        h_pool5_flat = tf.reshape(h_pool5, [-1, DEPTH_5])

        ############################################### DROPOUT ##########################################################
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_pool5_flat, keep_prob)

        ############################################# SOFTMAX OUTPUT LAYER ###############################################
        W_fc2 = weight_variable([DEPTH_5, NUM_OUTPUTS])
        b_fc2 = bias_variable([NUM_OUTPUTS])

        y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        ##################################### SETTING UP THE OPTIMISATION PROBLEM #####################################
        cross_entropy = -tf.reduce_sum(y_*tf.log(y+1e-12))
        loss = cross_entropy + mu*(tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2)
                                    + tf.nn.l2_loss(W_conv3) + tf.nn.l2_loss(W_conv4)
                                    + tf.nn.l2_loss(W_conv5) + tf.nn.l2_loss(W_fc2))

        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        xentrop_summ = tf.scalar_summary("Cross Entropy", loss)
        accuracy_summ = tf.scalar_summary("Accuracy", accuracy)


    return graph, train_step, x, y_, keep_prob, loss, accuracy, xentrop_summ, accuracy_summ

@ex.automain
def train(START_ITER, ITERS, RUN_NAME, MODEL_LOAD_PATH, DATA_AUGMENTATION, DIASTOLE, DATA_DIR,
        TRAIN_LABEL_NOISE_STD, TRAIN_LABEL_SMOOTHING_STD, NUM_REPS):
    (graph, train_step, x, y_, keep_prob,
        loss, accuracy, xentrop_summ, accuracy_summ) = construct_graph()

    train,train_labels,test,test_labels = load_data()

    lboard = np.load(os.path.join(DATA_DIR, 'standardized_lboard.npy'))
    lboard_labels = np.load(os.path.join(DATA_DIR, 'labels_lboard.npy'))

    train = np.concatenate((train, lboard), axis=1)
    train_labels = np.concatenate((train_labels, lboard_labels), axis=0)

    if DIASTOLE:
        train = train[:,::2,:,:,:,:]
        test = test[:,::2,:,:,:,:]
        train_labels = train_labels[::2]
        test_labels = test_labels[::2]
    ###############################################################################

    with tf.Session(graph=graph) as session:

        saver = tf.train.Saver(max_to_keep=100)

        if MODEL_LOAD_PATH:
            saver.restore(session, MODEL_LOAD_PATH)
        else:
            print 'Initializing'
            session.run(tf.initialize_all_variables())

        train_crps = []
        test_crps = []
        ensure_dir_exists(RUN_NAME)
        writer = tf.train.SummaryWriter(os.path.join(RUN_NAME, 'logs'),
                                        session.graph_def)

        for i in range(START_ITER,START_ITER+ITERS):
            Xt,yt = minibatch(train, train_labels, 200, noise_std=TRAIN_LABEL_NOISE_STD)

            if DATA_AUGMENTATION:
                Xt = augmentation(Xt)

            yt = make_class_label(yt)

            _,summ_str, summ_str2 = session.run([train_step,xentrop_summ, accuracy_summ],
                                      feed_dict={x:Xt,
                                                 y_: yt,
                                                 keep_prob: 0.25})
            writer.add_summary(summ_str, i)
            writer.add_summary(summ_str2, i)

            if i % 1000 == 0:
                print 'Validating'
                print 'Saved', saver.save(session,
                                          os.path.join(RUN_NAME, 'model'),
                                          global_step=i)

                Xv,yv = minibatch(test, test_labels, 2000, noise_std=TRAIN_LABEL_NOISE_STD)
                yv = make_class_label(yv)

                validation_score = session.run([accuracy], feed_dict={x: Xv, y_: yv, keep_prob: 1.0})

                # mbsize = NUM_REPS*num_pats
                # mb_scores = []
                # for patient in range(200):
                #     for perm in range(144):
                #         session.run([accuracy], feed_dict={x: Xv, y_: yv, keep_prob: 1.0})

                # for p,v in zip(range(0,Ly,num_pats),range(0,Lx,mbsize)):
                #     mb_scores.append(session.run([mean_crps],
                #                        feed_dict={x:Xv[v:min(Lx,v+mbsize)],
                #                                 y_:yv[p:min(Ly,p+num_pats)],
                #                                 weights: wv[p:min(Ly,p+num_pats)],
                #                                 keep_prob:1})[0])

                with open(os.path.join(RUN_NAME, 'logfile'), 'a') as out_file:
                    out_file.write('%.8f\n' % validation_score[0])



