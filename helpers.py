import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import dicom
import pylab
import numpy as np
from glob import glob
import cv2
from scipy import ndimage
import scipy
import scipy.stats
from matplotlib import animation
from JSAnimation.IPython_display import display_animation
import pickle

# make a file named local_paths.py and declare path in root_train
from local_paths import root_train

GAUSSIAN_BLUR_PRE_CENTROID_STFD = 3.
GET_LV_GAUSSIAN_BLUR = 3.
WEIGHTED_PROB_MASK_GAUSSIAN_BLUR = 3.
CONNECTED_COMPONENTS_MEDIAN_BLUR = 3.
CONNECTED_COMPONENTS_CONST_THRESH = 0
MAX_AREA_PROP = 1.
MAX_ML = 50
BBOX_ITERS = 1

def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def animate(examples, cmap=pylab.cm.bone, save=False):

    fig = plt.figure()
    im = plt.imshow(examples[0], cmap=cmap)

    # function to update figure
    def updatefig(j):
        data = examples[j]
        im.set_data(data)
        im.set_extent([0,data.shape[0],0,data.shape[1]])
        return im,

    # kick off the animation
    ani = animation.FuncAnimation(fig, updatefig, frames=len(examples),
                                  interval=50, blit=True)
    if save is not False:
        ani.save(save, writer='imagemagick', fps=10)
    return display_animation(ani, default_mode='loop')

def interleave(*mats):

    num_things = len(mats)

    s = list(np.shape(mats[0]))
    s[0] *= num_things

    result = np.zeros(s)
    for i,d in enumerate(mats):
        result[i::num_things] = d

    return result

def dash(t):
    m,n,y,x = t.shape
    return t.swapaxes(1, 2).reshape(m*y, n*x)

def line_mask(result,row,col):
    x = np.zeros_like(result)
    x[row,:] = result.max()
    x[:,col] = result.max()
    x = cv2.GaussianBlur(x, (3, 3), 0)
    return x

def ind2vec(ind, mat):

    return np.unravel_index(ind, np.shape(mat))

def correct(p):

    y_min, y_max = 0, 1
    w,c = p.WindowWidth, p.WindowCenter
    x = p.pixel_array
    corrected_x = np.ones(x.shape)*np.nan

    corrected_x[(x <= c - 0.5 - (w-1)/2)] = y_min

    corrected_x[(x > c - 0.5 + (w-1)/2)] = y_max

    corrected_x[np.isnan(corrected_x)] = ((x[np.isnan(corrected_x)] - (c - 0.5)) / (w-1) + 0.5) * (y_max - y_min )+ y_min 
    return corrected_x

def get_pics(dicom_array):
    return map(lambda x: correct(x), dicom_array)

def all_times(dirpath):
    pics = os.listdir(dirpath)
    times = []
    for path in sorted(pics):
        times.append(dicom.read_file(os.path.join(dirpath, path)))
    return times


def all_slices(case=1, pix=False, root=root_train):
    datadir = os.path.join(root, '%d/study/' % case)
    slices = []
    sizes = []
    lengths = []

    meta = (
        'PixelSpacing',
        'SliceThickness',
        'SliceLocation',
        'PatientAge',
        'PatientSex'
        # add field names we want
    )
    meta_defaults = {
        'PixelSpacing': (1.5*1.5),
        'SliceThickness': 8.,
        'PatientAge': 30,
        'PatientSex': 'M'
        # 'SliceLocation':
        # add defaults we want
    }

    def parse_age(agestring):
        if agestring[-1] == 'Y':
            return int(agestring[:-1])
        if agestring[-1] == 'M':
            return int(agestring[:-1])/12.
        if agestring[-1] == 'W':
            return int(agestring[:-1])/52.
    meta_funcs = {
        'PixelSpacing': lambda (h,w): float(h)*float(w),
        'SliceThickness': float,
        'SliceLocation': float,
        'PatientAge': parse_age,
        'PatientSex': lambda x: x,
        # add lambdas
    }


    def get_meta(sli):
        sli_meta = {}
        for att in meta:
            try:
                val = meta_funcs[att](getattr(sli[0], att))
            except AttributeError:
                val = meta_defaults[att]
                print 'Warning: Patient %d has a missing value for %s.' % (case, att)
                print 'Using default: %s' % str(val)
            sli_meta[att] = val
        return sli_meta

    for path in sorted(glob(os.path.join(datadir, 'sax_*')),
                       key=lambda x: int(x.split('sax_')[-1])):
        slices.append(all_times(path))
        lengths.append(len(slices[-1]))
        layer_sizes = set([np.shape(t.pixel_array) for t in slices[-1]])
        assert len(layer_sizes) == 1, 'Slice has more than one image size'
        sizes.append(layer_sizes.pop())

    size_counts = [(sizes.count(s),s) for s in sizes]
    count, most_common_size = sorted(size_counts, reverse=True)[0]
    if len(size_counts) > 1:
        print 'Warning: Patient %d has different sized images.' % case
        print 'Using most common size %s. Remaining %d images.' % (most_common_size,count)
        keep = np.array([s==most_common_size for s in sizes])
        lengths = np.array(lengths)[keep]
        slices = np.array(slices)[keep]

    common_length = min(lengths)
    if len(lengths) > 1:
        print 'Warning: Patient %d has more than one series length.' % case
        print 'Using min length %d.' %  common_length


    new_slices = []
    prev = -500
    prev_idx = -1 # holds the index of the previous element in slices
    for idx,sli in sorted(enumerate(slices), key=lambda (i,x): x[0].SliceLocation):
        cur = sli[0].SliceLocation
        sli = sli[:common_length]
        sli_meta = get_meta(sli)

        if pix:
            sli = get_pics(sli)


        if (cur - prev < 1) and (idx>prev_idx):
            new_slices[-1] = (sli, sli_meta)
        else:
            new_slices.append((sli, sli_meta))

        prev = cur
        prev_idx = idx

    return zip(*new_slices)

def spatial_temporal_finite_difference(x):
    return np.max(x,0)-np.min(x,0)

def threshold_std(img,std=1):

    return img*(img>(img.mean()+std*img.std()))

def fix_contrast(imgs):
    ''' imgs has dimensions (slices, time-frames, rows, cols)'''
    max_along_cols = np.max(np.reshape(imgs, (imgs.shape[0],imgs.shape[1],imgs.shape[2]*imgs.shape[3])), 2)
    max_along_rows = np.max(max_along_cols, 1)
    thresh = np.min(max_along_rows)
    return imgs*(imgs<thresh)

def bounding_square(result, row, col, incs=5, prop=.85):
    rsize, csize = result.shape
    width = 1
    condition = True
    result = result.astype(float)
    total = result.sum()
    while condition:

        cand = result[max(0,row-width):min(row+width,rsize-1),
                      max(0,col-width):min(col+width, csize-1)].sum()/float(total)
        condition = (cand < prop)
        # if not ((row-width > 0) and (row+width < rsize) and \
        #                 (col-width > 0) and (col+width < csize)):
        #     print 'Size restriction'
        #     print row-width, row+width, col-width, col+width, cand
        #     break

        width += incs
    return (max(0,row-width),min(row+width,rsize-1) ,
                max(0,col-width), min(col+width, csize-1))

def get_centroid(result):
    total = np.sum(result)
    return (int(sum([row*v for (row,v) in \
                         enumerate(np.sum(result,1))])//total),
            int(sum([col*v for (col,v) in \
                         enumerate(np.sum(result,0))])//total))

def matrix_values_in(matrix, array):
    res = np.zeros_like(matrix)
    for i,val in enumerate(array):
        res += (i+1)*(matrix == val)
    return res

def filter_components(tmask):
    '''Filters the connected components in tmask and returns metadata.'''

    times = []
    keep = []
    groups = np.unique(tmask)
    gareas = []
    gcentroids = []
    for g in groups:
        gmask = tmask==g
        area = np.sum(gmask)
        if area < MAX_AREA_PROP*tmask.size:
            keep.append(g)
            gareas.append(area)
            gcentroids.append(get_centroid(gmask))

    return matrix_values_in(tmask, keep), gareas, gcentroids

def masks_to_contours(tmask):
    '''Converts connected components to contours defined by their convex hulls'''
    tmask = tmask.astype('uint8')
    # assert tmask.sum()>0, tmask.sum()
    contours = np.zeros_like(tmask)
    for comp in np.unique(tmask):
        if comp == 0:
            continue
        r,c = (tmask==comp).nonzero()
        points = np.array(zip(c,r)) #Clean this up
        hull = cv2.convexHull(points)
        contour = np.zeros_like(tmask)
        cv2.drawContours(contour, [hull], 0, 1, thickness=-1)
        contours += comp*(contour)

    return contours

def circularity(mask):
    r,c = mask.nonzero()

    radius = max(max(r)-min(r), max(c)-min(c))/2.
    if radius == 0:
        return 1e-20
    return mask.sum()/(np.pi*(radius**2))



def weighted_probability_mask(mask, gaussian_blur=WEIGHTED_PROB_MASK_GAUSSIAN_BLUR):
    comps = np.unique(mask)

    if len(comps)<2: # whole thing is one component
        return np.ones_like(mask)/float(mask.size)

    sum_scores = 0
    scores = []
    masks = []
    for comp in comps:
        if comp == 0:
            continue
        comp_mask = mask==comp
        score = circularity(comp_mask)
        sum_scores += score
        scores.append(score)
        masks.append(comp_mask)

    scores = np.array(scores)/sum_scores
    mask = np.sum(scores[:,None, None] * np.array(masks), 0)
    mask += 1e-10
    mask = ndimage.gaussian_filter(mask, sigma=gaussian_blur)
    return mask/mask.sum()

def get_lv(modes, mask, blur_radius=GET_LV_GAUSSIAN_BLUR):
    modes = ndimage.gaussian_filter(modes, blur_radius)
    groups = np.unique(mask)
    scores = []
    for g in groups:
        gmask = mask == g
        scores.append(modes[gmask].mean())
    ind = np.argmax(scores)
    m = groups[ind]

    if m == 0:
        return np.zeros_like(mask)
    return mask==m

def prop_through_time(masks, prior):
    for t in range(len(masks)):
        mask = masks[t]
        posterior = weighted_probability_mask(mask)*prior
        posterior /= posterior.sum()
        # posterior = ndimage.gaussian_filter(posterior, 1.)
        estimate = get_lv(posterior, mask)
        yield prior, posterior, estimate
        prior = posterior

def calculate_layer_prior(track, prev):

    prior = np.mean(track[prev, :],0) + 1e-10
    prior /= prior.sum()

    return prior

def exag(img):
    blurred_f = ndimage.gaussian_filter(img, CONNECTED_COMPONENTS_GAUSSIAN_BLUR)
    edges = (img - blurred_f)
    negedges = edges < edges.max()/2
    return ndimage.median_filter(img*negedges, CONNECTED_COMPONENTS_MEDIAN_BLUR)

def get_slice_components(sli, perc):
    components = []
    for time in sli:
        # get slice components
        # time = exag(time)
        fimg = time > np.percentile(time, q=perc)
        tmask, num_comps = ndimage.label(fimg)
        components.append(tmask)

    #Unpack the components tuple
    return components



def linear_interpolate(areas, num_points = 30):
    num_layers = areas.shape[0]
    a_interp = []
    for i in range(num_layers):
        xp = areas[i].nonzero()[0]
        if xp.size>0:
            fp = areas[i,xp]
            x = np.arange(num_points)
            a_interp.append(np.interp(x, xp, fp))
        else:
            a_interp.append(np.zeros(num_points,))
    return a_interp

def get_volumes_from_track(track, pix_spacing, heights):
    # sum over pixels (still have (num_layers, num_times) array)
    a = track.sum(2).sum(2)
    num_times = track.shape[1]
    a_interp = linear_interpolate(a, num_times)

    # multiply each layer by its pixel size to get mm^2
    areas = pix_spacing[:,None] * a_interp

    # multiply each area by its height to get mm^3
    volumes = heights[:,None] * areas

    # return the sum of volumes over layers in ml
    return np.array(volumes)/1000.

def get_heights_from_meta(meta):
    locs = np.array([m['SliceLocation'] for m in meta])
    heights = np.zeros_like(locs)
    heights[1:-1] = (locs[2:] - locs[:-2])/2.
    heights[0] = heights[-1] = np.mean(heights[1:-1])
    return heights

def fourier_interpolation(curve):
    ft = scipy.fft(curve)
    ft[3:-2] *= 0.2
    return np.abs(scipy.ifft(ft))

def get_ef_from_volume(vol):
    return float(np.max(vol)-np.min(vol))/np.max(vol)

def crps(cdf, actual_vol):
    '''Takes a CDF ranging from 0 to 599mL of our estimate of an LV volume, and returns its CRPS'''
    return np.mean(np.array([(cdf[i] - float((i >= actual_vol)))**2 for i in range(len(cdf))]))

def is_bad_curve(curve):
    if (np.mean(curve[:4]) - np.mean(curve[9:15])) < 0:
        return True
    ft = scipy.fft(curve)
    return (np.linalg.norm(ft[3:-2])/np.linalg.norm(ft[1:]))**2 > 0.1

def run_pipeline(case, RUN_NAME=None, root=root_train):
    patient, meta = all_slices(case, pix=True, root=root)

    # get layer meta data
    pix_spacing = np.array([m['PixelSpacing'] for m in meta])
    heights = get_heights_from_meta(meta)

    # fix contrast
    # patient = fix_contrast(np.array(patient))

    # get middle
    num_layers, num_times = np.shape(patient)[0:2]
    mid_layer = num_layers//2

    ############### start get region of interest ##################
    for iteration in range(BBOX_ITERS):
        mid_slice = patient[mid_layer]
        mid_slice_stfd = spatial_temporal_finite_difference(mid_slice)
        mid_slice_stfd = threshold_std(mid_slice_stfd)
        # blur
        result = ndimage.gaussian_filter(mid_slice_stfd, GAUSSIAN_BLUR_PRE_CENTROID_STFD)
        # get centroid
        row,col = get_centroid(result)
        # get roi
        rstart, rend, cstart, cend = bounding_square(result, row, col)
        # crop patient
        patient = map(lambda sl: map(lambda x: x[rstart:rend, cstart:cend], sl), patient)

    ############### end get region of interest ##################

    ## GET EXPECTED SIZE AS PROP OF ROI ##
    max_size_pix = 1000.*MAX_ML/(heights.min()*pix_spacing.min())
    roi_size_pix = ((rend-rstart)*(cend-cstart))
    lv_fraction = 1.*max_size_pix/roi_size_pix

    frac_to_remove = max(.7, min(.8, 1.-lv_fraction))*100
    print frac_to_remove
    ## END GET EXPECTED SIZE AS PROP OF ROI ##

    ############### start prop through space and time ##################
    # initialize tracks array
    shape = np.shape(patient)
    tracks = np.zeros(shape)
    priors = np.zeros(shape)
    posts = np.zeros(shape)
    masks = np.zeros(shape)


    # compute mid_layer prior and prop through time
    mid_layer_stfd = spatial_temporal_finite_difference(patient[mid_layer])
    prior = ndimage.gaussian_filter(mid_layer_stfd, GAUSSIAN_BLUR_PRE_CENTROID_STFD**2) #TODO: update this

    half = num_times//2

    masks[mid_layer] = get_slice_components(patient[mid_layer], perc=frac_to_remove)
    priors[mid_layer, :half], posts[mid_layer, :half], tracks[mid_layer, :half] = \
        zip(*prop_through_time(masks[mid_layer, :half], prior=prior))
    rev = prop_through_time(masks[mid_layer, half:][::-1], prior=prior)
    priors[mid_layer, half:], posts[mid_layer, half:], tracks[mid_layer, half:] = zip(*list(rev)[::-1])
    # prop down
    for curr in range(mid_layer+1,num_layers):
        prev = curr - 1
        prior = calculate_layer_prior(tracks, prev=prev)
        masks[curr] = get_slice_components(patient[curr], perc=frac_to_remove)
        priors[curr, :half], posts[curr, :half], tracks[curr, :half] = \
            zip(*prop_through_time(masks[curr, :half], prior=prior))
        rev = prop_through_time(masks[curr, half:][::-1], prior=prior)
        priors[curr, half:], posts[curr, half:], tracks[curr, half:] = zip(*list(rev)[::-1])
    # prop up
    for curr in range(mid_layer)[::-1]: # mid_layer-1, mid_layer-2
        prev = curr + 1
        prior = calculate_layer_prior(tracks, prev=prev)
        masks[curr] = get_slice_components(patient[curr], perc=frac_to_remove)
        priors[curr], posts[curr], tracks[curr] = \
            zip(*prop_through_time(masks[curr], prior=prior))
        rev = prop_through_time(masks[curr, half:][::-1], prior=prior)
        priors[curr, half:], posts[curr, half:], tracks[curr, half:] = zip(*list(rev)[::-1])


    ############### end prop through space and time ##################

    del patient # save memory

    volumes = get_volumes_from_track(tracks, pix_spacing, heights)
    volume = np.sum(volumes, 0)

    min_vol,max_vol = min(volume), max(volume)


    if RUN_NAME is not None:
        # SAVE STUFF #
        masks /= masks.max(2).max(2)[:,:,None,None]
        priors /= priors.max(2).max(2)[:,:,None,None]
        posts /= posts.max(2).max(2)[:,:,None,None]
        tracks /= tracks.max(2).max(2)[:,:,None,None]


        dirname = os.path.join(RUN_NAME, '%d' % case)
        ensure_dir_exists(dirname)


        conc_dash = interleave(masks,priors,posts,tracks)
        # with open(os.path.join(dirname, 'track.npy'), 'w') as output:
        #     np.save(output, conc_dash)
        with open(os.path.join(dirname, 'volumes.npy'), 'w') as output:
            np.save(output, volumes)
        with open(os.path.join(dirname, 'meta.pkl'), 'w') as output:
            pickle.dump(meta, output)

        plt.imsave(os.path.join(dirname, 'track.png'), dash(conc_dash))
        # END SAVE STUFF #

    return min_vol, max_vol

if __name__=='__main__':

    import pandas
    train_labels = pandas.read_csv(root_train+'.csv')
    train_labels['EF'] = (train_labels.Diastole - train_labels.Systole)/train_labels.Diastole
    train_labels = train_labels.set_index(train_labels.Id)

    # train_labels = train_labels.drop([41,83,148,195,222,260,280,305,337,393,437,442,456])

    import sys
    try:
        num_cases = int(sys.argv[1])
    except:
        num_cases = len(train_labels)
    try:
        RUN_NAME = sys.argv[2]
    except:
        import datetime
        RUN_NAME = str(datetime.datetime.now())

    import random
    random.seed(2345)
    rows = random.sample(train_labels.index, num_cases)

    batch = train_labels.loc[rows]
    batch['Min'], batch['Max'] = -1.,-1.

    import time
    start = time.time()
    # mins = []
    # maxs = []
    for i,case in enumerate(batch['Id']):
        try:
            if os.path.exists(os.path.join(RUN_NAME, str(case))):
                continue
            batch['Min'][case], batch['Max'][case] = run_pipeline(case,RUN_NAME)
        except Exception as e:
            print 'Patient %d broken' % case
            print e

        if (i % 25) == 0:
            print 'Done with %d' % i

    end = time.time()
    batch['EF Estimate'] = (batch['Max'] - batch['Min'])/batch['Max']
    print batch
    batch.to_csv(os.path.join(RUN_NAME, 'predictions.csv'))

    print 'Pearson Correlation Diastole:', batch.corr()['Diastole']['Max']
    print 'RMSE Diastole:', np.sqrt(np.mean(np.square(batch['Max']-batch['Diastole'])))
    print 'Pearson Correlation Systole:', batch.corr()['Systole']['Min']
    print 'RMSE Systole:', np.sqrt(np.mean(np.square(batch['Min']-batch['Systole'])))
    print 'Pearson Correlation EF:', batch.corr()['EF']['EF Estimate']
    print 'RMSE EF:', np.sqrt(np.mean(np.square(batch['EF']-batch['EF Estimate'])))
    print 'Time Elapsed:', end-start



