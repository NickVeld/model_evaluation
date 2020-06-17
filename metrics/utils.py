import sklearn.metrics

import metrics


def get_metric(metric_name):
    if metric_name.startswith('sklearn.'):
        metric = getattr(sklearn.metrics, metric_name[metric_name.find('.') + 1:])
    else:
        metric = getattr(metrics, metric_name)()
    metric.metric_name = metric_name
    return metric
