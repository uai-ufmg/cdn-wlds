import numpy as np

'''
"Scaling" can be considered as "applying constant noise to the entire samples" whereas 
"Jittering" can be considered as "applying different noise to each sample".
"Magnitude Warping" can be considered as "applying smoothly-varing noise to the entire samples"
'''


def Jitter(X, columns_used, sigma=0.05):
    """
    Arguments:
    ---------
        - X (np.array): Well Log Dataset
        - columns_used (list[str]): List of logs used (GR, NPHI, ...)
        - sigma (float): Standard devitation (STD) of the jitter noise
    Return:
    ---------
        - (np.array): Data with jitter applied
    """
    myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X+myNoise


def Scaling(X, columns_used, sigma=0.1):
    """
    Arguments:
    ---------
        - X (np.array): Well Log Dataset
        - columns_used (list[str]): List of logs used (GR, NPHI, ...)
        - sigma (float): Standard devitation (STD) of distribution that generates the scaling factor
    Return:
    ---------
        - (np.array): Data with scaling applied
    """
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1,X.shape[1])) # #shape 1,len(instance) - allows broadcasting
    myNoise = np.matmul(np.ones((X.shape[0],1)), scalingFactor) #acho que isso gera uma matix 1x1 - matmul aqui vai dar 1,X.shape[1] vs X.shape[0],1
    return X*myNoise


## This example using cubic splice is not the best approach to generate random curves. 
## You can use other approaches, e.g., Gaussian process regression, Bezier curve, etc.
def GenerateRandomCurves(X, columns_used, sigma=0.1, knot=4):
    """
    Arguments:
    ---------
        - X (np.array): Well Log Dataset
        - columns_used (list[str]): List of logs used (GR, NPHI, ...)
        - sigma (float): Standard devitation (STD) of distribution that helps generate the random curve for magnitude warp
        - knot (int): number of knots for the random curves (Complexity of the curves)
    Return:
    ---------
        - (np.array): Random Cubic Spline curve 
    """

    from scipy.interpolate import CubicSpline  

    xx = (np.ones((X.shape[1],1))*(np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    
    cs_all_columns = []
    for i in range(len(columns_used)):
        cs_all_columns.append(CubicSpline(xx[:,i], yy[:,i]))
        
    return np.array([cs_all_columns[i](x_range) for i in range(len(columns_used))]).transpose()


def MagWarp(X, columns_used, sigma=0.1, knot=4):
    """
    Arguments:
    ---------
        - X (np.array): Well Log Dataset
        - columns_used (list[str]): List of logs used (GR, NPHI, ...)
        - sigma (float): Standard devitation (STD) of distribution that helps generate the random curve for magnitude warp
        - knot (int): number of knots for the random curves (Complexity of the curves)
    Return:
    ---------
        - (np.array): Data multiplied by the random curve to generate the magnitude warp
    """

    return X * GenerateRandomCurves(X, columns_used, sigma=sigma, knot=knot)


def DistortTimesteps(X, columns_used, sigma=0.1, knot=4):
    """
    Arguments:
    ---------
        - X (np.array): Well Log Dataset
        - columns_used (list[str]): List of logs used (GR, NPHI, ...)
        - sigma (float): Standard devitation (STD) of distribution that helps generate the random curve for time warp
        - knot (int): number of knots for the random curves (Complexity of the curves)
    Return:
    ---------
        - (np.array): time warp factor
    """

    tt = GenerateRandomCurves(X, columns_used, sigma, knot=knot) # Regard these samples aroun 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)        # Add intervals to make a cumulative graph
    # Make the last value to have X.shape[0]
    t_scale = [(X.shape[0]-1)/tt_cum[-1,i] for i in range(len(columns_used))]
    for i in range(len(columns_used)):
        tt_cum[:,i] = tt_cum[:,i]*t_scale[i]
        
    return tt_cum


def TimeWarp(X, columns_used, sigma=0.2, knot=4):
    """
    Arguments:
    ---------
        - X (np.array): Well Log Dataset
        - columns_used (list[str]): List of logs used (GR, NPHI, ...)
        - sigma (float): Standard devitation (STD) of distribution that helps generate the random curve for time warp
        - knot (int): number of knots for the random curves (Complexity of the curves)
    Return:
    ---------
        - (np.array): Well log data interpolated with time warp factor to generate a time-warped Well Log data
    """

    tt_new = DistortTimesteps(X, columns_used, sigma,knot=knot)
    X_new = np.zeros(X.shape)
    x_range = np.arange(X.shape[0])
    for i in range(len(columns_used)):
        X_new[:,i] = np.interp(x_range, tt_new[:,i], X[:,i])
        
    return X_new
