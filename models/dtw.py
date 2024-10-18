from dtw import *

def dtw_distance(x, y, dist='euclidean'):
    alignment = dtw(x, y, dist_method=dist, distance_only=True)
    return alignment.distance