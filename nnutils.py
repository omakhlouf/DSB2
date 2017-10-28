import numpy as np
from scipy import ndimage

def minibatch(dat, labels, size=500, noise_std=1e-8):
    '''Return random minibatch of shape (size,32,32,3)'''
    num_groups, _, _, num_times, rsize, csize = np.shape(dat)
    case_inds = np.random.randint(0, len(labels), size=size)

    data = np.zeros((size,rsize,csize,9))

    gsize = num_times//3
    time_inds_1 = np.random.randint(0, gsize, size=size)
    time_inds_2 = np.random.randint(gsize, 2*gsize, size=size)
    time_inds_3 = np.random.randint(2*gsize, num_times, size=size)

    data[:,:,:,0] = dat[0][case_inds,np.random.randint(0,2,size), time_inds_1]
    data[:,:,:,1] = dat[1][case_inds,np.random.randint(0,2,size), time_inds_1]
    data[:,:,:,2] = dat[2][case_inds,np.random.randint(0,2,size), time_inds_1]

    data[:,:,:,3] = dat[0][case_inds,np.random.randint(0,2,size), time_inds_2]
    data[:,:,:,4] = dat[1][case_inds,np.random.randint(0,2,size), time_inds_2]
    data[:,:,:,5] = dat[2][case_inds,np.random.randint(0,2,size), time_inds_2]

    data[:,:,:,6] = dat[0][case_inds,np.random.randint(0,2,size), time_inds_3]
    data[:,:,:,7] = dat[1][case_inds,np.random.randint(0,2,size), time_inds_3]
    data[:,:,:,8] = dat[2][case_inds,np.random.randint(0,2,size), time_inds_3]

    y = labels[case_inds] + np.random.normal(0, noise_std, size) # perhaps do this better


    return data, y.reshape((-1,1))

def batch(dat, labels):
    '''Takes dat as a (num_groups, num_cases, slices_per_group, num_times,
    rsize, csize) tensor and returns (num_cases * num_combs, rsize, csize,
    num_groups) where num_combs is slices_per_group**num_groups and each
    case is represented by '''

    num_slice_groups, num_cases, slices_per_group, num_times, rsize, csize = np.shape(dat)
    #num_cases, slices_per_group, num_times, rsize, csize = np.shape(dat)
    # there are 8 combinations of slices, and 8 time steps
    num_time_groups = 3
    num_groups = num_time_groups*num_slice_groups

    num_slice_combs = (slices_per_group**num_slice_groups)
    num_time_combs = 18
    num_combs = num_time_combs*num_slice_combs

    data = np.zeros((num_cases*num_combs,rsize,csize,num_groups), dtype='float16')
    labels = np.repeat(labels, num_combs)

    slice_perms = [(i,j,k) for i in range(slices_per_group)
                    for j in range(slices_per_group)
                    for k in range(slices_per_group)]
    time_perms = [(i,j,k) for i in range(num_times//3)
                    for j in range(num_times//3,2*num_times//3)
                    for k in range(2*num_times//3,num_times)]
    comb = 0
    for (i,j,k) in slice_perms:
        for (x,y,z) in time_perms:
            data[comb::num_combs,:,:,0] = dat[0][:, i, x]
            data[comb::num_combs,:,:,1] = dat[1][:, j, x]
            data[comb::num_combs,:,:,2] = dat[2][:, k, x]
            #
            data[comb::num_combs,:,:,3] = dat[0][:, i, y]
            data[comb::num_combs,:,:,4] = dat[1][:, j, y]
            data[comb::num_combs,:,:,5] = dat[2][:, k, y]
            #
            data[comb::num_combs,:,:,6] = dat[0][:, i, z]
            data[comb::num_combs,:,:,7] = dat[1][:, j, z]
            data[comb::num_combs,:,:,8] = dat[2][:, k, z]
            comb += 1

    return data, labels.reshape((-1,1))

def get_random_in_range(start,end):
    rng = end - start
    return start + rng*np.random.random()

from skimage import transform
def augmentation(dat):
    num_cases = np.shape(dat)[0]
    for i in range(num_cases):
        swirl_stength = get_random_in_range(-.4,.4)
        dat[i] = transform.swirl(dat[i], strength=swirl_stength)
        rotation_angle = get_random_in_range(-30,30)
        dat[i] = transform.rotate(dat[i], angle=rotation_angle)
        if np.random.random() > .5:
            dat[i] = np.flipud(dat[i])
    return dat

def rescale_data_and_labels(dat, labels):
    rescaled_dat = np.zeros_like(dat)
    rescaled_labels = np.zeros_like(labels)
    for i in range(len(dat)):
        y = labels[i][0]
        X = dat[i]

        factor = min(599./y,2**get_random_in_range(-1,1))

        rescaled_labels[i] = factor*y
        rescaled_dat[i] = scale_img_preserve_shape(X,factor)

    return rescaled_dat, rescaled_labels

def scale_img_preserve_shape(img, scale):
    scale = np.sqrt(scale)
    shape = np.array(img.shape[:2])
    img = img.reshape((shape[0],shape[1],-1))
    scaled = transform.rescale(img,scale)
    new_shape = scaled.shape[:2]
    if scale < 1.:

        pad = np.max([(0,0), shape - new_shape], 0)/2.

        pad = ((int(np.ceil(pad[0])), int(np.floor(pad[0]))),
               (int(np.ceil(pad[1])), int(np.floor(pad[1]))),
               (0,0))

        scaled = np.pad(
            scaled,
            pad,
            mode='constant', constant_values=0)

    else:
        #Readjust the shape of the resized image
        rbound = np.floor((scale-1)*shape[0]/2.)
        cbound = np.floor((scale-1)*shape[1]/2.)
        scaled = scaled[rbound:rbound+shape[0],
                        cbound:cbound+shape[1]]
    return scaled

def median_filter(X):
    medians = np.median(np.median(X,1), 1)
    return X*(X>medians[:,None,None,:])

def scale_unit_var(X):
    mean = np.mean(np.mean(X,1), 1)[:,None,None,:]
    std = np.std(np.std(X,1), 1)[:,None,None,:]
    return (X-mean)/std

def make_cdf(y, std=0.):
    '''Takes a list of numpy array of point estimates 
    and returns a (N,600) step function CDF'''

    y = np.array(y).astype(int)
    pdf = np.zeros((y.shape[0],600))
    for i in range(y.shape[0]):
        pdf[i,y[i]] = 1.
        pdf[i] = ndimage.gaussian_filter1d(pdf[i], std)
        pdf[i] /= pdf[i].sum()
    cdf = np.cumsum(pdf, 1)
    cdf[cdf>1.] = 1.

    return cdf

def average_repeated_predictions(test_preds, num_reps=64):
    '''Takes (num_cases*num_reps, output_size) and returns (num_cases,
    output_size) by averaging over the num_reps.'''

    num_points, output_size = np.shape(test_preds)
    return np.reshape(test_preds,(-1, num_reps, output_size)).mean(1)

if __name__== '__main__':
    train = np.load('netdata/standardized_train_3232.npy')
    labels = np.load('netdata/labels_train.npy')

    batchdata,batchlabels = minibatch(train, labels)
    print batchdata.shape
