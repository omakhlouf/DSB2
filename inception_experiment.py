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

def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

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
    NUM_INPUTS = 9       # Number of input channels
    NUM_OUTPUTS = 600    # Number of output classes in the softmax layer
    NUM_REPS = 144

    DEPTH_1 = 4                            # The output depth of the first inception module
    DEPTH_2 = 4                            # The output depth of the second inception module
    DEPTH_3 = 4                            # The output depth of the third inception module
    DEPTH_4 = 4                            # The output depth of the fourth inception module
    DEPTH_5 = 8                            # The output depth of the fifth inception module
    DEPTH_6 = 8                            # The output depth of the sixth inception module
    NUM_OUT1 = 3*DEPTH_1 + NUM_INPUTS      # Number of output channels after first inception
    NUM_OUT2 = 3*DEPTH_2 + NUM_OUT1        # Number of output channels after second inception
    NUM_OUT3 = 3*DEPTH_3 + NUM_OUT2        # Number of output channels after third inception
    NUM_OUT4 = 3*DEPTH_4 + NUM_OUT3        # Number of output channels after fourth inception
    NUM_OUT5 = 3*DEPTH_5 + NUM_OUT4        # Number of output channels after fifth inception
    NUM_OUT6 = 3*DEPTH_6 + NUM_OUT5        # Number of output channels after sixth inception

    mu = 0.0001
    LEARNING_RATE = 1e-4

    REGULARIZE_BIAS = False

    TRAIN_LABEL_NOISE_STD = 0.01
    TRAIN_LABEL_SMOOTHING_STD = 0.01
    DATA_AUGMENTATION = True
    ENLARGED = False
    SHIFT = False
    DIASTOLE = False

    DATA_MEDIAN_FILTER = False
    WEIGHTS_PATH = None

@ex.named_config
def inception_exp():
    mu = 0.00001
    LEARNING_RATE = 4e-4
    RUN_NAME = 'INCEP-EXP3'

@ex.named_config
def inception_exp_2():
    mu = 0.0001
    LEARNING_RATE = 8e-4
    RUN_NAME = 'INCEP-EXP4'

@ex.named_config
def inception_exp_3():
    mu = 0.0002
    LEARNING_RATE = 8e-4
    RUN_NAME = 'INCEP-EXP5'
    NUM_INPUTS = 9       # Number of input channels
    NUM_OUTPUTS = 600    # Number of output classes in the softmax layer
    NUM_REPS = 144
    DEPTH_1 = 4                            # The output depth of the first inception module
    DEPTH_2 = 4                            # The output depth of the second inception module
    DEPTH_3 = 4                            # The output depth of the third inception module
    DEPTH_4 = 4 
    DEPTH_5 = 4
    DEPTH_6 = 4
    NUM_OUT1 = 3*DEPTH_1 + NUM_INPUTS      # Number of output channels after first inception
    NUM_OUT2 = 3*DEPTH_2 + NUM_OUT1        # Number of output channels after second inception
    NUM_OUT3 = 3*DEPTH_3 + NUM_OUT2        # Number of output channels after third inception
    NUM_OUT4 = 3*DEPTH_4 + NUM_OUT3
    NUM_OUT5 = 3*DEPTH_5 + NUM_OUT4
    NUM_OUT6 = 3*DEPTH_6 + NUM_OUT5 

@ex.named_config
def inception_exp_4():
    mu = 0.0001
    LEARNING_RATE = 2e-4
    RUN_NAME = 'INCEP-EXP6'
    NUM_INPUTS = 9       # Number of input channels
    NUM_OUTPUTS = 600    # Number of output classes in the softmax layer
    NUM_REPS = 144
    DEPTH_1 = 4                            # The output depth of the first inception module
    DEPTH_2 = 4                            # The output depth of the second inception module
    DEPTH_3 = 4                            # The output depth of the third inception module
    DEPTH_4 = 4 
    DEPTH_5 = 4
    DEPTH_6 = 4
    NUM_OUT1 = 3*DEPTH_1 + NUM_INPUTS      # Number of output channels after first inception
    NUM_OUT2 = 3*DEPTH_2 + NUM_OUT1        # Number of output channels after second inception
    NUM_OUT3 = 3*DEPTH_3 + NUM_OUT2        # Number of output channels after third inception
    NUM_OUT4 = 3*DEPTH_4 + NUM_OUT3
    NUM_OUT5 = 3*DEPTH_5 + NUM_OUT4
    NUM_OUT6 = 3*DEPTH_6 + NUM_OUT5 

@ex.named_config
def inception_exp_depth():
    mu = 0.0001
    LEARNING_RATE = 2e-4
    RUN_NAME = 'INCEP-EXP-DEP'
    NUM_INPUTS = 9       # Number of input channels
    NUM_OUTPUTS = 600    # Number of output classes in the softmax layer
    NUM_REPS = 144
    DEPTH_1 = 8                            # The output depth of the first inception module
    DEPTH_2 = 8                            # The output depth of the second inception module
    DEPTH_3 = 4                            # The output depth of the third inception module
    DEPTH_4 = 4 
    DEPTH_5 = 4
    DEPTH_6 = 4
    NUM_OUT1 = 3*DEPTH_1 + NUM_INPUTS      # Number of output channels after first inception
    NUM_OUT2 = 3*DEPTH_2 + NUM_OUT1        # Number of output channels after second inception
    NUM_OUT3 = 3*DEPTH_3 + NUM_OUT2        # Number of output channels after third inception
    NUM_OUT4 = 3*DEPTH_4 + NUM_OUT3
    NUM_OUT5 = 3*DEPTH_5 + NUM_OUT4
    NUM_OUT6 = 3*DEPTH_6 + NUM_OUT5 

@ex.named_config
def inception_exp_full():
    mu = 0.0001
    LEARNING_RATE = 2e-4
    RUN_NAME = 'INCEP-EXP-FULL'
    NUM_INPUTS = 9       # Number of input channels
    NUM_OUTPUTS = 600    # Number of output classes in the softmax layer
    NUM_REPS = 144
    DEPTH_1 = 8                            # The output depth of the first inception module
    DEPTH_2 = 10                            # The output depth of the second inception module
    DEPTH_3 = 20                            # The output depth of the third inception module
    DEPTH_4 = 24 
    DEPTH_5 = 30
    DEPTH_6 = 30

@ex.capture
def load_data(DATA_DIR):
    train = np.load(os.path.join(DATA_DIR, 'standardized_train.npy'))
    train_labels = np.load(os.path.join(DATA_DIR, 'labels_train.npy'))
    test = np.load(os.path.join(DATA_DIR, 'standardized_test.npy'))
    test_labels = np.load(os.path.join(DATA_DIR, 'labels_test.npy'))


    return train, train_labels, test, test_labels

@ex.capture
def inception(x, NUM_INPUTS, NUM_OUTPUTS):
    '''An Inception module with an output depth of (4*NUM_OUTPUTS)'''
    #### a 1x1 convolution
    W_conv1 = weight_variable([1, 1, NUM_INPUTS, NUM_OUTPUTS])
    b_conv1 = bias_variable([NUM_OUTPUTS])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1) ### FIRST OUTPUT
    
    #### a 3x3 convolution, first reducing the dimensionality using a 1x1 convolution
    W_conv2 = weight_variable([1, 1, NUM_INPUTS, NUM_OUTPUTS/2])
    b_conv2 = bias_variable([NUM_OUTPUTS/2])
    h_conv2 = tf.nn.relu(conv2d(x, W_conv2) + b_conv2)

    W_conv3= weight_variable([3, 3, NUM_OUTPUTS/2, NUM_OUTPUTS])
    b_conv3= bias_variable([NUM_OUTPUTS])
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3) ### SECOND OUTPUT
    
    #### a 5x5 convolution, first reducing the dimensionality using a 1x1 convolution
    W_conv4 = weight_variable([1, 1, NUM_INPUTS, NUM_OUTPUTS/2])
    b_conv4 = bias_variable([NUM_OUTPUTS/2])
    h_conv4 = tf.nn.relu(conv2d(x, W_conv4) + b_conv4)

    W_conv5 = weight_variable([5, 5, NUM_OUTPUTS/2, NUM_OUTPUTS])
    b_conv5 = bias_variable([NUM_OUTPUTS])
    h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5) ### THIRD OUTPUT
    
    #### A parallel max pool branch
    h_pool1 = max_pool_3x3(x) ### FOURTH OUTPUT
    W_conv6 = weight_variable([1, 1, NUM_INPUTS, NUM_OUTPUTS])
    b_conv6 = bias_variable([NUM_OUTPUTS])
    h_conv6 = tf.nn.relu(conv2d(h_pool1, W_conv6) + b_conv6)

    #### Filter concatenation for final output
    y = tf.concat(3, [h_conv1, h_conv3, h_conv5, h_conv6])

    loss = tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(W_conv3) + tf.nn.l2_loss(W_conv4) + tf.nn.l2_loss(W_conv5) + tf.nn.l2_loss(W_conv6)

    return y, loss

@ex.capture
def construct_graph(PIC_WIDTH,DEPTH_1,DEPTH_2,DEPTH_3,DEPTH_4,DEPTH_5,DEPTH_6,
                    NUM_INPUTS,NUM_OUTPUTS,mu,LEARNING_RATE,NUM_REPS,REGULARIZE_BIAS):
    graph = tf.Graph()
    with graph.as_default():
        ####################################### INPUT/OUTPUT PLACEHOLDERS ##############################################
        x = tf.placeholder(tf.float32, shape=[None, PIC_WIDTH, PIC_WIDTH, NUM_INPUTS]) #Placeholder for the input images
        y_ = tf.placeholder(tf.float32, shape=[None, NUM_OUTPUTS]) #Placeholder for the label cdfs
        

        ################################# FIRST INCEPTION MODULE (No pooling) ########################################
        h_inception1, l1 = inception(x, NUM_INPUTS, DEPTH_1)
        ################################## SECOND INCEPTION MODULE (pooling) #########################################
        h_incept, l2 = inception(h_inception1, 4*DEPTH_1, DEPTH_2)
        h_inception2 = max_pool_2x2(h_incept)
        ################################# THIRD INCEPTION MODULE (no pooling) ########################################
        h_inception3, l3 = inception(h_inception2, 4*DEPTH_2, DEPTH_3)
        ################################### FOURTH INCEPTION MODULE (pooling) ########################################
        h_incept, l4 = inception(h_inception3, 4*DEPTH_3, DEPTH_4)
        h_inception4 = max_pool_2x2(h_incept)
        ################################# FIFTH INCEPTION MODULE (No pooling) ########################################
        h_inception5, l5 = inception(h_inception4, 4*DEPTH_4, DEPTH_5)
        #h_inception1 = max_pool_2x2(h_incept)
        ################################## SIXTH INCEPTION MODULE (pooling) #########################################
        h_incept, l6 = inception(h_inception5, 4*DEPTH_5, DEPTH_6)
        h_inception6 = max_pool_2x2(h_incept)

        ################################## Average Pooling on the last output #########################################
        h_pool6 = tf.nn.avg_pool(h_inception6,ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
        h_pool6_flat = tf.reshape(h_pool6, [-1, 4*DEPTH_6])

        ############################################### DROPOUT ##########################################################
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_pool6_flat, keep_prob)

        ############################################# SOFTMAX OUTPUT LAYER ###############################################
        W_fc2 = weight_variable([4*DEPTH_6, NUM_OUTPUTS])
        b_fc2 = bias_variable([NUM_OUTPUTS])

        y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        y = cumsum(y_conv)
        ##################################### SETTING UP THE OPTIMISATION PROBLEM #####################################
        crps = tf.reduce_mean(tf.reduce_mean(tf.square(y - y_), 1)) #y_ here should be the label cdfs
        ### Could add learning rate decay here - 1e-4 is the current learning rate
        
        loss = crps + mu*(l1 + l2 + l3 + l4 + l5 + l6+ tf.nn.l2_loss(W_fc2))

        if REGULARIZE_BIAS:
            loss += mu*(tf.nn.l2_loss(b_fc2))

        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

        mean_y = tf.reduce_mean(tf.reshape(y,(-1, NUM_REPS, NUM_OUTPUTS)), 1)
        mean_crps = tf.reduce_mean(tf.reduce_mean(tf.square(mean_y - y_), 1))

        mean_crps_summ = tf.scalar_summary("mean crps", mean_crps)
        crps_summ = tf.scalar_summary("crps", crps)


    return graph, train_step, x, y_, keep_prob, crps, mean_y, mean_crps, crps_summ, mean_crps_summ

@ex.command
def submission(MODEL_LOAD_PATH, DATA_DIR, OUT_DIR, NUM_REPS):

    testdat = np.load(os.path.join(DATA_DIR, 'standardized_lboard.npy'))
    graph, _, x, _, keep_prob, _, mean_y, _, _, _ = construct_graph()

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

@ex.command
def train_all(START_ITER, ITERS, RUN_NAME, MODEL_LOAD_PATH, DATA_AUGMENTATION, DATA_MEDIAN_FILTER,
        TRAIN_LABEL_NOISE_STD, TRAIN_LABEL_SMOOTHING_STD, NUM_REPS):
    (graph, train_step, x, y_, keep_prob,
        _, _, mean_crps, crps_summ, mean_crps_summ,weights) = construct_graph()

    train,train_labels,test,test_labels = load_data()

    if WEIGHTS_PATH is not None:
        train_weights = np.load(os.path.join(WEIGHTS_PATH, 'weights_train.npy'))
        test_weights = np.load(os.path.join(WEIGHTS_PATH, 'weights_test.npy'))
    else:
        train_weights = np.ones_like(train_labels)
        test_weights = np.ones_like(test_labels)

    train = np.concatenate((train, test), axis=1)
    train_labels = np.concatenate((train_labels, test_labels), axis=0)
    train_weights = np.concatenate((train_weights, test_weights), axis=0)

    with tf.Session(graph=graph) as session:

        saver = tf.train.Saver(max_to_keep=100)

        if MODEL_LOAD_PATH:
            saver.restore(session, MODEL_LOAD_PATH)
        else:
            print 'Initializing'
            session.run(tf.initialize_all_variables())


        ensure_dir_exists(RUN_NAME)
        writer = tf.train.SummaryWriter(os.path.join(RUN_NAME, 'logs'),
                                        session.graph_def)

        for i in range(START_ITER,START_ITER+ITERS):

            Xt,yt,wt = minibatch(train, train_labels, 100, noise_std=TRAIN_LABEL_NOISE_STD, weights=train_weights)

            if DATA_MEDIAN_FILTER:
                Xt = median_filter(Xt)
            if DATA_AUGMENTATION:
                Xt = augmentation(Xt)

            yt = make_cdf(yt, std=TRAIN_LABEL_SMOOTHING_STD)
            _,summ_str = session.run([train_step,crps_summ],
                                      feed_dict={x:Xt,
                                                 y_: yt,
                                                 weights: wt,
                                                 keep_prob: 0.5})
            writer.add_summary(summ_str, i)

            if i % 1000 == 0:
                print 'Saved', saver.save(session,
                                          os.path.join(RUN_NAME, 'model'),
                                          global_step=i)


@ex.automain
def train(START_ITER, ITERS, RUN_NAME, MODEL_LOAD_PATH, DATA_AUGMENTATION, ENLARGED, SHIFT, DIASTOLE, WEIGHTS_PATH,
    DATA_MEDIAN_FILTER,TRAIN_LABEL_NOISE_STD, TRAIN_LABEL_SMOOTHING_STD, NUM_REPS):
    (graph, train_step, x, y_, keep_prob,
        _, _, mean_crps, crps_summ, mean_crps_summ) = construct_graph()

    train,train_labels,test,test_labels = load_data()
    if DIASTOLE:
        train = train[:,::2,:,:,:,:]
        test = test[:,::2,:,:,:,:]
        train_labels = train_labels[::2]
        test_labels = test_labels[::2]
    ################################ Experimental #################################
    # train = train[:, train_labels>150, :,:,:,:]
    # train_labels = train_labels[train_labels > 150]
    # test = test[:, test_labels>150, :,:,:,:]
    # test_labels = test_labels[test_labels>150]
    ###############################################################################
    if WEIGHTS_PATH is not None:
        train_weights = np.load(os.path.join(WEIGHTS_PATH, 'weights_train.npy'))
        test_weights = np.load(os.path.join(WEIGHTS_PATH, 'weights_test.npy'))
    else:
        train_weights = np.ones_like(train_labels)
        test_weights = np.ones_like(test_labels)

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
        yv = make_cdf(test_labels)
        wv = test_weights.reshape((-1,1))

        Lx = len(Xv)
        Ly = len(yv)

        for i in range(START_ITER,START_ITER+ITERS):
            Xt,yt,_ = minibatch(train, train_labels, 100, noise_std=TRAIN_LABEL_NOISE_STD, weights=train_weights)
            if DATA_MEDIAN_FILTER:
                Xt = median_filter(Xt)
            if DATA_AUGMENTATION:
                Xt = augmentation(Xt)
            if ENLARGED:
                Xt, yt = enlarge(Xt, yt)
            if SHIFT:
                Xt, yt = shift(Xt, yt)

            yt = make_cdf(yt, std=TRAIN_LABEL_SMOOTHING_STD)
            _,summ_str = session.run([train_step,crps_summ],
                                      feed_dict={x:Xt,
                                                 y_: yt,
                                                 keep_prob: 0.5})
            writer.add_summary(summ_str, i)

            if i % 1000 == 0:
                print 'Validating'
                print 'Saved', saver.save(session,
                                          os.path.join(RUN_NAME, 'model'),
                                          global_step=i)
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



