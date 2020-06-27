import numpy as np
import sklearn.metrics

from metrics.base import _metric_base

class ME(_metric_base):
    '''
    Mean error
    '''

    TRANSFORM_NAME = "Error"
    SHORT_TRANSFORM_NAME = "E"

    # If __call__ is redifined
    # __call__.__doc__ = _metric_base.__doc__

    def transform(self, y_true, y_pred):
        return y_pred - y_true

class MAE(_metric_base):
    '''
    Mean absolute error
    '''

    TRANSFORM_NAME = "Absolute error"
    SHORT_TRANSFORM_NAME = "AE"

    def transform(self, y_true, y_pred):
        return np.abs(y_pred - y_true)


class MAPE(_metric_base):
    '''
    Mean absolute percentage error

    NOT for ~zero values!
    '''
    
    TRANSFORM_NAME = "Absolute percentage error"
    SHORT_TRANSFORM_NAME = "APE"

    def transform(self, y_true, y_pred):
        return np.abs(1 - y_pred / y_true)


class MASE(_metric_base):
    '''
    Mean absolute scaled error
    '''
    def __init__(self, *args, **kwargs):
        pass

    TRANSFORM_NAME = "Absolute scaled error"
    SHORT_TRANSFORM_NAME = "ASE"

    def transform(self, y_true, y_pred):
        return (np.abs(y_true - y_pred)
                / np.mean(np.abs(y_true[1:] - y_true[:-1])))

class MASPE(_metric_base):
    '''
    Mean absolute scaled over predicted error
    '''
    def __init__(self, *args, **kwargs):
        pass

    TRANSFORM_NAME = "Absolute scaled over predicted error"
    SHORT_TRANSFORM_NAME = "ASPE"

    def transform(self, y_true, y_pred):
        return (np.abs(y_true - y_pred)
                / np.mean(np.abs(y_pred[1:] - y_pred[:-1])))

class MALE(_metric_base):
    '''
    Mean absolute logarithmic error
    '''
    def __init__(self, *args, **kwargs):
        pass

    TRANSFORM_NAME = "Absolute logarithmic error"
    SHORT_TRANSFORM_NAME = "ALE"

    def transform(self, y_true, y_pred):
        return np.abs(np.log((y_pred + 1) / (y_true + 1)))
