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
