
# coding: utf-8

# In[1]:
import matplotlib
matplotlib.use('Agg')
from helpers import *
import sys
import time



def generate_gifs(case):
    start = time.time()

    patient = all_slices(case, pix=True)
    #Get the middle slice
    num_layers = len(patient)
    mid_layer = num_layers//2
    mid_slice = patient[mid_layer]



    # # Phase 1 - Data Cleanup

    # ## 1.1 - Fixing High Contrast

    # In[4]:

    patient = fix_contrast(np.array(patient))
    mid_slice = patient[mid_layer]



    # ## 1.2 - Find the area that changes most in middle slice

    # In[5]:

    mid_slice_stfd = spatial_temporal_finite_difference(mid_slice)


    # ### 1.2.2 - Threshold at a value to cleanup a bit more

    # In[6]:

    mid_slice_stfd = threshold_std(mid_slice_stfd,1.)



    # ## 1.3 - Get the centroid of this region

    # In[7]:

    # blur
    result = cv2.GaussianBlur(mid_slice_stfd, (9, 9), 0)
    # get centroid
    row,col = get_centroid(result)
    center = line_mask(result, row, col)


    # ## 1.4 - Find a good bounding square around the centroid to reduce the size of the image to a smaller ROI

    # In[8]:

    # get roi
    rstart, rend, cstart, cend = bounding_square(result, row, col)
    width = rend-rstart

    patient = map(lambda sli: map(lambda x: x[rstart:rend, cstart:cend], sli), patient)
    old_mid_slice_stfd = mid_slice_stfd.copy()
    mid_slice_stfd = mid_slice_stfd[rstart:rend,cstart:cend]

    print 'Size reduction:', ((rend-rstart)**2)/float(result.size)
    rlm = line_mask(result, rstart, cstart)
    clm = line_mask(result, rend, cend)
    center = line_mask(result, row, col)


    # # Phase 2 - Segmentation

    # ## 2.1 - Get most significant connected components for the mid-slice over each time-frame

    # In[9]:
    i = np.argmax(ndimage.gaussian_filter(mid_slice_stfd, 9))
    p = np.zeros((mid_slice_stfd.size,))
    p[i] = 1
    p = ndimage.gaussian_filter(p.reshape(mid_slice_stfd.shape),15)
    p/=p.sum()



    # In[10]:

    track = np.zeros((num_layers,30,width, width))


    # # 2.2 - Start with mid layer and prop components through time and space



    masks,areas,centroids =  get_slice_components(patient, mid_layer)
    modes = prop_through_time(masks, p)
    track[mid_layer, :] = [lv for lv in prop_through_time(masks, p,30)]



    # In[16]:

    for i in range(1,mid_layer+1):
        prop_through_space(track, patient, mid_layer, i)
    for i in range(1,num_layers-mid_layer):
        prop_through_space(track, patient, mid_layer, -i)

    end = time.time()

    print 'Done! %2.f seconds elapsed.' % (end-start)

    # In[17]:
    dirname = '%d' % case
    ensure_dir_exists(dirname)
    np.save(os.path.join(dirname, 'track.npy'), track)
    for i in range(num_layers):
        print 'Generating output for %d'%i
        fname = os.path.join(dirname, '%d.gif' % i)
        animate(map(lambda x: track[i,x]*1000.+patient[i][x], range(30)), None, save=fname)
        plt.figure()
        plt.plot(track[i,:,:,:].sum(1).sum(1))
        plt.savefig(os.path.join(dirname, '%d_area_curve.png' % i))
        plt.close()

    plt.figure()
    plt.plot(track.sum(0).sum(1).sum(1))
    plt.close()
    plt.savefig(os.path.join(dirname, 'volume_curve.png'))

if __name__ == '__main__':
    
    cases = [1,13]
    broken = []
    for case in cases:
        try:
            print 'Case #%d'%case
            generate_gifs(case)
        except Exception as e:
            broken.append(case)
            print 'Broken', case
    print broken

