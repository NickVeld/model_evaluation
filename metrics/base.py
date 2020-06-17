class _metric_base:
    '''
    Metric name

    Some doctring
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
        return 0.
