def make_pdf(cdf):
    cdf = np.pad(cdf, pad_width=((0,0), (1,0)),mode='constant', constant_values=0)
    assert np.all(cdf[:,0] == 0)
    return np.diff(cdf, 1)
