import numpy as np
import sklearn.metrics

from metrics.base import _metric_base

class MAE(_metric_base):
    '''
    Mean absolute error
    '''

    def __call__(self, y_true, y_pred):
        return sklearn.metrics.mean_absolute_error(y_true, y_pred)
    __call__.__doc__ = _metric_base.__doc__


class MAPE(_metric_base):
    '''
    Mean absolute percentage error

    NOT for ~zero values!
    '''

    def __call__(self, y_true, y_pred):
        #if (y_true == 0).any() or (y_pred == 0).any():
        #    raise ValueError("MAPE does not work with zeros!")
        return np.mean(np.abs(1 - y_pred/y_true))
    __call__.__doc__ = _metric_base.__doc__


class MASE:
    '''
    Mean absolute scaled error
    '''
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, y_true, y_pred):
        '''
        :param y_true: actual values
        :type y_true: array-like of shape = (n_samples)
        :param y_pred: predicted values
        :type y_pred: array-like of shape = (n_samples)
        :return: Metric value
        :rtype: float or np.float64
        '''
        return (np.mean(np.abs(y_true - y_pred))
                / np.mean(np.abs(y_true[1:] - y_true[:-1])))


class MALE:
    '''
    Mean absolute logarithmic error
    '''
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, y_true, y_pred):
        '''
        :param y_true: actual values
        :type y_true: array-like of shape = (n_samples)
        :param y_pred: predicted values
        :type y_pred: array-like of shape = (n_samples)
        :return: Metric value
        :rtype: float or np.float64
        '''
        return np.mean(np.abs(np.log((y_pred + 1) / (y_true + 1))))
