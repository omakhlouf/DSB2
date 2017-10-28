import tensorflow as tf
import numpy as np
from nnutils import *
# from helpers import ensure_dir_exists
import pandas, os
import time

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

#from joblib import Parallel, delayed
#from joblib import load, dump
#import tempfile
def aug(x):
    x.astype('float32')
    swirl_stength = get_random_in_range(-.4,.4)
    x = transform.swirl(x, strength=swirl_stength)
    rotation_angle = get_random_in_range(-30,30)
    x = transform.rotate(x, angle=rotation_angle)
    if np.random.random() > .5:
        x = np.flipud(x)
    return x.astype('float16')

def aug_row(input, output, i):
    """Compute the sum of a row in input and store it in output"""
    output[i] = aug(input[i])

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
    DEPTH_1 = 20         # The output depth of the first convolutional layer
    DEPTH_2 = 40         # The output depth of the second convolutional layer
    DEPTH_3 = 80         # The output depth of the second convolutional layer
    DEPTH_4 = 150        # The output depth of the second convolutional layer
    DEPTH_5 = 150        # The output depth of the second convolutional layer
    DEPTH_6 = 150        # The output depth of the second convolutional layer
    NUM_HIDDEN = 400     # Number of hidden units in the hidden layer
    NUM_OUTPUTS = 600    # Number of output classes in the softmax layer
    KERNEL_X = 3         # The width of the convolution kernel (using same for 1st and 2nd layers)
    KERNEL_Y = 3         # The height of the convolution kernel (using same for 1st and 2nd layers)
    mu = 0.0001
    LEARNING_RATE = 1e-4

    REGULARIZE_BIAS = False


    NUM_INPUTS = 3       # Number of input channels
    NUM_REPS = 64
    MBSIZE = 100

    TRAIN_LABEL_NOISE_STD = 2.
    TRAIN_LABEL_SMOOTHING_STD = 0.
    DATA_AUGMENTATION = False

    DATA_MEDIAN_FILTER = False
    DATA_UNIT_VAR = False
    VAL_PER = 1000
    DROPOUT = 0.5
    ALSO_MIN_RMSE = False
    TEMP_DIR_PATH = '.'
    TRAIN_ALL = False
    FILTER_BY_LABEL = False

@ex.named_config
def over_80():
    NUM_INPUTS = 9       # Number of input channels
    NUM_REPS = 144
    TRAIN_LABEL_NOISE_STD = 1.
    TRAIN_LABEL_SMOOTHING_STD = 2.
    DATA_AUGMENTATION = True
    RUN_NAME = 'OVER-80'
    FILTER_BY_LABEL = True
    LEARNING_RATE = 1e-3
    MBSIZE = 2000
    VAL_PER = 200

@ex.named_config
def under_110():
    NUM_INPUTS = 9       # Number of input channels
    NUM_REPS = 144
    TRAIN_LABEL_NOISE_STD = 1.
    TRAIN_LABEL_SMOOTHING_STD = 2.
    DATA_AUGMENTATION = True
    RUN_NAME = 'UNDER-110'
    FILTER_BY_LABEL = True
    LEARNING_RATE = 1e-3
    MBSIZE = 2000
    VAL_PER = 200

@ex.capture
def load_data(DATA_DIR):
    train = np.load(os.path.join(DATA_DIR, 'standardized_train.npy'))
    train_labels = np.load(os.path.join(DATA_DIR, 'labels_train.npy'))
    test = np.load(os.path.join(DATA_DIR, 'standardized_test.npy'))
    test_labels = np.load(os.path.join(DATA_DIR, 'labels_test.npy'))

    return train, train_labels, test, test_labels

@ex.capture
def construct_graph(PIC_WIDTH,DEPTH_1,DEPTH_2,DEPTH_3,DEPTH_4,DEPTH_5,DEPTH_6,
                    NUM_HIDDEN,NUM_INPUTS,NUM_OUTPUTS,KERNEL_X,KERNEL_Y,mu,
                    LEARNING_RATE,NUM_REPS,REGULARIZE_BIAS):
    graph = tf.Graph()
    with graph.as_default():
        ####################################### INPUT/OUTPUT PLACEHOLDERS ##############################################
        x = tf.placeholder(tf.float32, shape=[None, PIC_WIDTH, PIC_WIDTH, NUM_INPUTS]) #Placeholder for the input images
        y_ = tf.placeholder(tf.float32, shape=[None, NUM_OUTPUTS]) #Placeholder for the label cdfs
        y_pests = tf.placeholder(tf.float32, shape=[None])
        lr = tf.placeholder(tf.float32)
        ####################################### FIRST CONVOLUTIONAL LAYER ##############################################
        # The weight tensor has dimensions [kernel_size_x, kernel_size_y, num_input_channels, num_output_channels]
        W_conv1 = weight_variable([KERNEL_X, KERNEL_Y, NUM_INPUTS, DEPTH_1])
        b_conv1 = bias_variable([DEPTH_1])

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
        h_pool5_flat = tf.reshape(h_pool5, [-1, DEPTH_6])

        ############################################### DROPOUT ##########################################################
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_pool5_flat, keep_prob)

        ############################################# SOFTMAX OUTPUT LAYER ###############################################
        W_fc2 = weight_variable([DEPTH_5, NUM_OUTPUTS])
        b_fc2 = bias_variable([NUM_OUTPUTS])

        y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        #Cumulative sum
        y = cumsum(y_conv)
        ##################################### SETTING UP THE OPTIMISATION PROBLEM #####################################
        l2 =  mu*(tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2)
                        + tf.nn.l2_loss(W_conv3) + tf.nn.l2_loss(W_conv4)
                        + tf.nn.l2_loss(W_conv5) + tf.nn.l2_loss(W_fc2))
        if REGULARIZE_BIAS:
            l2 += mu*(tf.nn.l2_loss(b_conv1) + tf.nn.l2_loss(b_conv2)
                    + tf.nn.l2_loss(b_conv3) + tf.nn.l2_loss(b_conv4)
                    + tf.nn.l2_loss(b_conv5) + tf.nn.l2_loss(b_fc2))

        labs = tf.constant(np.arange(600, dtype='float32').reshape((1,-1)))
        point = tf.reduce_sum(y_conv*labs, 1)
        rmse = tf.sqrt(tf.reduce_mean(tf.square(point - y_pests)))
        rmse_loss = rmse + l2
        rmse_train_step = tf.train.AdamOptimizer(lr).minimize(rmse_loss)

        crps = tf.reduce_mean(tf.square(y - y_)) #y_ here should be the label cdfs
        ### Could add learning rate decay here - 1e-4 is the current learning rate
        loss = crps + l2
        train_step = tf.train.AdamOptimizer(lr).minimize(loss)

        mean_y = tf.reduce_mean(tf.reshape(y,(-1, NUM_REPS, NUM_OUTPUTS)), 1)
        mean_crps = tf.reduce_mean(tf.square(mean_y - y_))

        mean_crps_summ = tf.scalar_summary("mean crps", mean_crps)
        crps_summ = tf.scalar_summary("crps", crps)


    return graph, train_step, x, y_, keep_prob, crps, mean_y, mean_crps, crps_summ, mean_crps_summ, rmse_train_step, rmse, y_pests, lr

@ex.command
def submission(MODEL_LOAD_PATH, DATA_DIR, OUT_DIR, NUM_REPS):

    testdat = np.load(os.path.join(DATA_DIR, 'standardized_lboard.npy'))
    graph, _, x, _, keep_prob, _, mean_y, _, _, _, _, _, _, _ = construct_graph()

    X, _ = batch(testdat, [])
    Lx = len(X)
    with tf.Session(graph=graph) as session:
        saver = tf.train.Saver()
        saver.restore(session, MODEL_LOAD_PATH)

        num_pats = 10
        mbsize = NUM_REPS*num_pats
        cdfs = []

        for v in range(0,Lx,mbsize):
            cdfs.extend(session.run([mean_y],
                                   feed_dict={x:X[v:min(Lx,v+mbsize)],
                                    keep_prob:1})[0])



    sub = prepare_submission(np.array(cdfs))
    sub.to_csv(os.path.join(OUT_DIR, 'submission.csv'))


@ex.automain
def train(START_ITER, ITERS, RUN_NAME, MODEL_LOAD_PATH, DATA_AUGMENTATION,
    DATA_MEDIAN_FILTER,TRAIN_LABEL_NOISE_STD, TRAIN_LABEL_SMOOTHING_STD,
    DATA_UNIT_VAR, NUM_REPS, MBSIZE, VAL_PER, ALSO_MIN_RMSE, TRAIN_ALL,
    LEARNING_RATE, TEMP_DIR_PATH, DROPOUT, FILTER_BY_LABEL, DATA_DIR):
    (graph, train_step, x, y_, keep_prob,
        crps, _, mean_crps, crps_summ, mean_crps_summ, rmse_train_step, rmse, y_pests, lr) = construct_graph()

    train,train_labels,test,test_labels = load_data()

    lboard = np.load(os.path.join(DATA_DIR, 'standardized_lboard.npy'))
    lboard_labels = np.load(os.path.join(DATA_DIR, 'labels_lboard.npy'))

    train = np.concatenate((train, lboard), axis=1)
    train_labels = np.concatenate((train_labels, lboard_labels), axis=0)

    del lboard, lboard_labels

    if FILTER_BY_LABEL:
        FILTER_BY_LABEL = lambda x: x<110#x>80
        train_inds = list(zip(*filter(lambda (i,x): FILTER_BY_LABEL(x), enumerate(train_labels)))[0])
        train = train[:,train_inds]
        train_labels = train_labels[train_inds]
        test_inds = list(zip(*filter(lambda (i,x): FILTER_BY_LABEL(x), enumerate(test_labels)))[0])
        test = test[:,test_inds]
        test_labels = test_labels[test_inds]

    if TRAIN_ALL:
        print 'TRAIN ALL'
        train = np.concatenate((train, test), axis=1)
        train_labels = np.concatenate((train_labels, test_labels), axis=0)

    with tf.Session(graph=graph) as session:

        saver = tf.train.Saver(max_to_keep=100)

        if MODEL_LOAD_PATH:
            print 'Loading', MODEL_LOAD_PATH
            saver.restore(session, MODEL_LOAD_PATH)
        else:
            print 'Initializing'
            session.run(tf.initialize_all_variables())

        ensure_dir_exists(RUN_NAME)
        writer = tf.train.SummaryWriter(os.path.join(RUN_NAME, 'logs'),
                                        session.graph_def)
        if not TRAIN_ALL:
            Xv, yv = batch(test, test_labels)

            if DATA_MEDIAN_FILTER:
                Xv = median_filter(Xv)
            if DATA_UNIT_VAR:
                Xv = scale_unit_var(Xv)
            yv = make_cdf(test_labels)
            Lx = len(Xv)
            Ly = len(yv)

        Xt, yt = batch(train_labels, train_labels)
        if DATA_MEDIAN_FILTER:
            Xt = median_filter(Xt)

        if DATA_AUGMENTATION:
            Xt = augmentation(Xt)

        else:
            Xt = Xt

        for i in range(START_ITER,START_ITER+ITERS):

            if i % VAL_PER == 0:
                print 'Validating'
                print 'Saved', saver.save(session,
                                          os.path.join(RUN_NAME, 'model'),
                                          global_step=i)
                if not TRAIN_ALL:
                    num_pats = 10
                    mbsize = NUM_REPS*num_pats
                    mb_scores = []

                    for p,v in zip(range(0,Ly,num_pats),range(0,Lx,mbsize)):
                        mb_scores.append(session.run([mean_crps],
                                           feed_dict={x:Xv[v:min(Lx,v+mbsize)],
                                                    y_:yv[p:min(Ly,p+num_pats)],
                                                    keep_prob:1})[0])
                    with open(os.path.join(RUN_NAME, 'logfile'), 'a') as out_file:
                        out_file.write('%.8f\n' % np.mean(mb_scores))

                if DATA_UNIT_VAR:
                    Xt = scale_unit_var(Xt)

                yt_pests = yt.flatten()
                yt = yt_pests + np.random.normal(loc=0,
                                            scale=TRAIN_LABEL_NOISE_STD,
                                            size=len(yt))
                yt = make_cdf(yt, std=TRAIN_LABEL_SMOOTHING_STD)


            inds = np.random.randint(0, len(Xt), size=MBSIZE)
            _,summ_str,c = session.run([train_step,crps_summ,crps],
                                      feed_dict={x:Xt[inds],
                                                 y_: yt[inds],
                                                 lr: LEARNING_RATE,
                                                 keep_prob: DROPOUT})
            writer.add_summary(summ_str, i)

            if ALSO_MIN_RMSE:
                start = time.time()
                _,r = session.run([rmse_train_step,rmse],
                          feed_dict={x:Xt[inds],
                                     y_pests: yt_pests[inds],
                                     lr: LEARNING_RATE/10.,
                                     keep_prob: 1.})
                elapsed =  time.time() - start
                print 'Elapsed %.5f, iter %d, rmse %.5f' % (elapsed, i, r)



