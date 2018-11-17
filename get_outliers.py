import numpy as np

def percentile_outliers(arr, upper, lower):
    Q1 = np.percentile(arr, 25)
    Q3 = np.percentile(arr, 75)
    IQR = Q3 - Q1
    threshold = []
    if upper==True:
        upper_threshold = Q3 + 1.5 * IQR
        threshold += [upper_threshold]
    if lower==True:
        lower_threshold = Q1 - 1.5 * IQR
        threshold += [lower_threshold]
    if len(threshold)==1: 
        threshold = threshold[0]
    return threshold