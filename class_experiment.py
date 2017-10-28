import tensorflow as tf
import numpy as np
from nnutils import *
# from helpers import ensure_dir_exists
import pandas, os

def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def prepare_submission(data):
    data = data.copy()
    data[data>1.] = 1.
    index = [val for pair in zip(['%d_Diastole' % case for case in range(501,701)],
                                  ['%d_Systole' % case for case in range(501,701)]) \
                 for val in pair]
    columns = ['P%d' % i for i in range(600)]
    return pandas.DataFrame(data=data, index=index, columns=columns)

def cumsum(softmax):
    values = tf.split(1, softmax.get_shape()[1], softmax)
    out = []
    prev = tf.zeros_like(values[1])
    for val in values:
        s = prev + val
        out.append(s)
        prev = s
    csum = tf.concat(1, out)
    return csum

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
#from sacred.observers import MongoObserver
ex = Experiment('DSB CLASS EXPERIMENT')
#ex.observers.append(MongoObserver.create())

@ex.config
def config():
    RUN_NAME = 'CRPS-MODEL-3.0'
    DATA_DIR = 'netdata'
    ITERS = 100000
    START_ITER = 0
    MODEL_LOAD_PATH = None
    PIC_WIDTH = 32
    ### Architectural Hyperparameters
    DEPTH_1 = 20         # The output depth of the first convolutional layer
    DEPTH_2 = 40         # The output depth of the second convolutional layer
    DEPTH_3 = 80         # The output depth of the second convolutional layer
    DEPTH_4 = 150        # The output depth of the second convolutional layer
    DEPTH_5 = 150        # The output depth of the second convolutional layer
    DEPTH_6 = 150        # The output depth of the second convolutional layer
    NUM_OUTPUTS = 2    # Number of output classes in the softmax layer
    KERNEL_X = 3         # The width of the convolution kernel (using same for 1st and 2nd layers)
    KERNEL_Y = 3         # The height of the convolution kernel (using same for 1st and 2nd layers)
    mu = 0.0001
    LEARNING_RATE = 1e-4

    REGULARIZE_BIAS = False
    TRAIN_LABEL_NOISE_STD = 0.0001
    TRAIN_LABEL_SMOOTHING_STD = 0.
    DATA_AUGMENTATION = True

    DATA_MEDIAN_FILTER = False
    NUM_INPUTS = 9       # Number of input channels
    NUM_REPS = 144
    TRAIN_LABEL_NOISE_STD = 1.
    TRAIN_LABEL_SMOOTHING_STD = 2.
    RUN_NAME = 'CLASS-EXP'

@ex.capture
def load_data(DATA_DIR):
    train = np.load(os.path.join(DATA_DIR, 'standardized_train.npy'))
    train_labels = np.load(os.path.join(DATA_DIR, 'labels_train.npy'))
    test = np.load(os.path.join(DATA_DIR, 'standardized_test.npy'))
    test_labels = np.load(os.path.join(DATA_DIR, 'labels_test.npy'))

    return train, train_labels, test, test_labels

@ex.capture
def conv_model(x,keep_prob,DEPTH_1,DEPTH_2,DEPTH_3,DEPTH_4,DEPTH_5,NUM_INPUTS,
                NUM_OUTPUTS,KERNEL_X,KERNEL_Y,mu,NUM_REPS,REGULARIZE_BIAS):
    ####################################### FIRST CONVOLUTIONAL LAYER ##############################################
    # The weight tensor has dimensions [kernel_size_x, kernel_size_y, num_input_channels, num_output_channels]
    W_conv1 = weight_variable([KERNEL_X, KERNEL_Y, NUM_INPUTS, DEPTH_1])
    b_conv1 = bias_variable([DEPTH_1])
    #Take the input image, reshape it to a 4D tensor with dimensions: [_, image_width, image_height, num_channels]
    #x_image = tf.reshape(x, [-1,32,32,3])
    # Convolve with kernel -> add bias -> apply ReLU -> max pool 
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
    W_conv5 = weight_variable([1, 1, DEPTH_4, DEPTH_5])
    b_conv5 = bias_variable([DEPTH_5])

    h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
    h_pool5 = tf.nn.avg_pool(h_conv5,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_pool5_flat = tf.reshape(h_pool5, [-1, DEPTH_5])
    ############################################### DROPOUT ##########################################################
    h_fc1_drop = tf.nn.dropout(h_pool5_flat, keep_prob)
    ############################################# SOFTMAX OUTPUT LAYER ###############################################
    W_fc2 = weight_variable([DEPTH_5, NUM_OUTPUTS])
    b_fc2 = bias_variable([NUM_OUTPUTS])

    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    loss =  mu*(tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2)
                    + tf.nn.l2_loss(W_conv3) + tf.nn.l2_loss(W_conv4)
                    + tf.nn.l2_loss(W_conv5) + tf.nn.l2_loss(W_fc2))
    if REGULARIZE_BIAS:
        loss += mu*(tf.nn.l2_loss(b_conv1) + tf.nn.l2_loss(b_conv2)
                + tf.nn.l2_loss(b_conv3) + tf.nn.l2_loss(b_conv4)
                + tf.nn.l2_loss(b_conv5) + tf.nn.l2_loss(b_fc2))

    return y, loss

@ex.capture
def construct_graph(PIC_WIDTH, NUM_INPUTS, NUM_OUTPUTS, LEARNING_RATE, NUM_REPS):

    graph = tf.Graph()
    with graph.as_default():
        ####################################### INPUT/OUTPUT PLACEHOLDERS ##############################################
        x = tf.placeholder(tf.float32, shape=[None, PIC_WIDTH, PIC_WIDTH, NUM_INPUTS]) #Placeholder for the input images
        y_ = tf.placeholder(tf.float32, shape=[None, 2]) #Placeholder for the label cdfs
        keep_prob = tf.placeholder(tf.float32)

        y, rg = conv_model(x, keep_prob, NUM_OUTPUTS)

        cross_entropy = -tf.reduce_sum(y_*tf.log(y))
        loss =  cross_entropy + rg

        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

        mean_y = tf.reduce_mean(tf.reshape(y,(-1, NUM_REPS, NUM_OUTPUTS)), 1)
        mean_cross = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y)))

        mean_crps_summ = tf.scalar_summary("mean cross entropy", mean_cross)
        crps_summ = tf.scalar_summary("cross entropy", cross_entropy)

        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        acc_summ = tf.scalar_summary("accuracy", accuracy)

    return graph, train_step, x, y_, keep_prob, cross_entropy, mean_y, mean_cross, crps_summ, mean_crps_summ, acc_summ


@ex.automain
def train(START_ITER, ITERS, RUN_NAME, MODEL_LOAD_PATH, DATA_AUGMENTATION,
    DATA_MEDIAN_FILTER,TRAIN_LABEL_NOISE_STD, TRAIN_LABEL_SMOOTHING_STD, NUM_REPS):
    (graph, train_step, x, y_, keep_prob,
        _, _, mean_cross, crps_summ, mean_crps_summ, acc_summ) = construct_graph()

    train, train_labels, test, test_labels = load_data()

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

        Xv, yv = batch(test, test_labels)
        if DATA_MEDIAN_FILTER:
            Xv = median_filter(Xv)

        ## Change labels for classification
        print 'Generating one hot labels'
        one_hot_labels = np.zeros((len(test_labels), 2))
        for i in range(len(test_labels)):
            if test_labels[i] > 240:
                one_hot_labels[i, 0] = 1
            else:
                one_hot_labels[i, 1] = 1
        print 'Got one hot labels'
        yv = one_hot_labels
        Lx = len(Xv)
        Ly = len(yv)

        for i in range(START_ITER,START_ITER+ITERS):
            Xt,yt, _ = minibatch(train, train_labels, size=50, noise_std=TRAIN_LABEL_NOISE_STD)
            one_hot_labels = np.zeros((len(yt), 2))
            for j in range(len(yt)):
                if yt[j] > 240:
                    one_hot_labels[j, 0] = 1
                else:
                    one_hot_labels[j, 1] = 1

            yt = one_hot_labels

            if DATA_MEDIAN_FILTER:
                Xt = median_filter(Xt)
            if DATA_AUGMENTATION:
                Xt = augmentation(Xt)
            
            #print 'Iteration', i
            #yt = make_cdf(yt, std=TRAIN_LABEL_SMOOTHING_STD)
            _,summ_str, acc_str = session.run([train_step, crps_summ, acc_summ],
                                      feed_dict={x:Xt,
                                                 y_: yt,
                                                 keep_prob: 0.5})
            writer.add_summary(summ_str, i)
            writer.add_summary(acc_str, i)

            if i % 1000 == 0:
            #     print 'Validating'
                print 'Saved', saver.save(session,
                                       os.path.join(RUN_NAME, 'model'),
                                    global_step=i)
            #     num_pats = 10
            #     mbsize = NUM_REPS*num_pats
            #     mb_scores = []

            #     for p,v in zip(range(0,Ly,num_pats),range(0,Lx,mbsize)):
            #         mb_scores.append(session.run([mean_cross],
            #                            feed_dict={x:Xv[v:min(Lx,v+mbsize)],
            #                                     y_:yv[p:min(Ly,p+num_pats)],
            #                                     keep_prob:1})[0])
            #     with open(os.path.join(RUN_NAME, 'logfile'), 'a') as out_file:
            #         out_file.write('%.8f\n' % np.mean(mb_scores))



