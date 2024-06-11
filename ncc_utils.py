import numpy as np


def convert_3d_to_linear(x,y,z,ny,nz):
    return z + y*nz + x*nz*ny

def convert_4d_to_linear(w,x,y,z,nx,ny,nz):
    return z + y*nz + x*nz*ny + w*nz*ny*nx

def load_test_data(rfd_path):
    from scipy.io import loadmat

    data = loadmat(rfd_path)
    rf = dict(arfi=data['arfiRf'], bmode=data['bmodeRf'])
    initialData = data['initialData']
    coords = data['coords']

    return rf, coords, initialData

def upsample_rf(rf, fsUpsampleFactor):
    from scipy import interpolate

    nAx = rf.shape[0]
    nAxUpsampled = nAx*fsUpsampleFactor

    x = np.linspace(0,1,nAx)
    xnew = np.linspace(0,1,nAxUpsampled)

    interp = interpolate.interp1d(x, rf, axis=0, kind='cubic')
    return interp(xnew).astype(np.float32)

def parabolicInterpolation(y1, y2, y3):
    val = y2 + (-y1*y1+2*y1*y3-y3*y3) / (8*y1-16*y2+8*y3)
    idx = (y1-y3) / (2*y1-4*y2+2*y3)
    return val, idx

def nthMoment(x, n, window_size):
    return np.convolve(x**n, np.ones((window_size,)), 'valid') / window_size