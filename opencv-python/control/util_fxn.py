import numpy as np

def load_mat(m_def):
    """Load a numpy matrix from dictionary."""
    mss = {}
    if 'dtype' in m_def:
        mss['dtype'] = m_def['dtype']
    if 'shape' in m_def:
        mss['shape'] = tuple(m_def['shape'])
    if 'data' in m_def:
        mss['buffer'] = np.array(m_def['data'])
    return np.ndarray(**mss)

def intersection(a,b):
    """ Calculate the intersection between two rectangles """
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0: return () # or (0,0,0,0) ?
    return (x, y, w, h)

def div0( a, b ):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c