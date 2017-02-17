import numpy as np

def load_mat(m_def):
    mss = {}
    if 'dtype' in m_def:
        mss['dtype'] = m_def['dtype']
    if 'shape' in m_def:
        mss['shape'] = tuple(m_def['shape'])
    if 'data' in m_def:
        mss['buffer'] = np.array(m_def['data'])
    return np.ndarray(**mss)

def intersection(a,b):
  x = max(a[0], b[0])
  y = max(a[1], b[1])
  w = min(a[0]+a[2], b[0]+b[2]) - x
  h = min(a[1]+a[3], b[1]+b[3]) - y
  if w<0 or h<0: return () # or (0,0,0,0) ?
  return (x, y, w, h)
