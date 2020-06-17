import inspect

import numpy as np
import sklearn.metrics

import metrics

y_true = np.array([2, 10, 20, 8])
y_pred = np.array([1, 5, 30, 12])
 

def test_metrics_workability():
    for metric_name, metric in inspect.getmembers(metrics, inspect.isclass):
        print(metric_name)
        if metric_name.startswith('_'):
            continue
        metric()(y_true, y_pred)
    print('sklearn.MAE')
    getattr(sklearn.metrics, 'mean_absolute_error')(y_true, y_pred)

def test_MAPE():
    assert getattr(metrics, 'MAPE')()(y_true, y_pred) == 0.5

def test_sklearn_mean_absolute_error():
    assert getattr(sklearn.metrics, 'mean_absolute_error')(y_true, y_pred) == 5