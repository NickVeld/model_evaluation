import pandas as pd
import numpy as np

from reporter import Reporter

import metrics


expected_merged_data_and_predictions = pd.DataFrame({
        'Confirmed': [1., 4., 22.],
        'Deaths': [np.nan, np.nan, np.nan],
        'Recovered': [np.nan, 1., np.nan],
        'Confirmed_1_Test': [5., np.nan, 11.],
        'Deaths_1_Test': [1., np.nan, 1.],
        'Recovered_1_Test': [np.nan, np.nan, 3.],
        'Confirmed_5_Test': [np.nan, 3., np.nan],
        'Deaths_5_Test': [np.nan, 4., np.nan],
        'Recovered_5_Test': [np.nan, np.nan, np.nan],
    }, index=pd.MultiIndex.from_tuples([
        ('Japan', '', pd.Timestamp('20200123')),
        ('Japan', '', pd.Timestamp('20200127')),
        ('Mainland China', 'Beijing', pd.Timestamp('20200123')),
        #('Mainland China', 'Beijing', pd.Timestamp('20200127')),
    ], names=('Country', 'Region', 'Date'))).sort_index()

expected_df3d = {'C' : pd.DataFrame({
        ('MAPE', '1') : [4., 0.5],
        ('MAPE', '5') : [0.25, np.nan],
        ('sklearn.mean_absolute_error', '1') : [4., 11.],
        ('sklearn.mean_absolute_error', '5') : [1., np.nan],
    }, index=pd.MultiIndex.from_tuples([
        ('Test', 'Japan'),
        ('Test', 'Mainland China/Beijing'),
    ], )).sort_index()}

def test_constructor():
    Reporter(output_path_template="output/test")

def test_get_locations_by_periods1():
    reporter = Reporter(output_path_template="output/test")
    prediction_index = np.array([
        ('Japan', '', '2020-01-23')
        , ('Japan', '', '2020-01-27')
        , ('Japan', '', '2020-01-29')
        , ('Mainland China', 'Beijing', '2020-01-23')
        , ('Mainland China', 'Beijing', '2020-01-27')
        , ('Mainland China', 'Beijing', '2020-01-29')
    ], dtype=object)

    start_dates, end_dates, countries, regions = reporter.get_locations_by_periods(prediction_index)

    assert start_dates == ['2020-01-23']
    assert end_dates == ['2020-01-29']
    assert countries == [{'Mainland China', 'Japan'}]
    assert regions == [{'', 'Beijing'}]

def test_get_locations_by_periods2():
    reporter = Reporter(output_path_template="output/test")
    prediction_index = np.array([
        ('Japan', '', '2020-01-23')
        , ('Japan', '', '2020-01-27')
        , ('Japan', '', '2020-01-29')
        , ('Mainland China', 'Beijing', '2020-01-23')
        , ('Mainland China', 'Beijing', '2020-01-27')
    ], dtype=object)

    start_dates, end_dates, countries, regions = reporter.get_locations_by_periods(prediction_index)

    assert start_dates == ['2020-01-23', '2020-01-29']
    assert end_dates == ['2020-01-27', '2020-01-29']
    assert countries == [{'Mainland China', 'Japan'}, {'Japan'}]
    assert regions == [{'', 'Beijing'}, {''}]

def test_prepare_data_and_prediction_for_report():
    reporter = Reporter(output_path_template="output/test")
    prediction_name = 'Test'

    merged_data_and_predictions = reporter.prepare_data_and_prediction_for_report([
        'predictions/' + prediction_name + '.csv'
    ])

    #print(merged_data_and_predictions.index.values, expected.index.values)
    #print(merged_data_and_predictions.columns.values, expected.columns.values)
    #print(merged_data_and_predictions)
    #print(expected)
    assert (expected_merged_data_and_predictions.index.values == merged_data_and_predictions.index.values).all()
    assert (expected_merged_data_and_predictions.columns.values == merged_data_and_predictions.columns.values).all()
    #print((merged_data_and_predictions == expected).all())
    assert (expected_merged_data_and_predictions.fillna(-1) == merged_data_and_predictions.fillna(-1)).all().all()
    #merged_data_and_predictions.to_csv(reporter.opt + ".csv")

def test_form_tables():
    reporter = Reporter(output_path_template="output/test")

    data_for_report = reporter.group_merged_data_and_predictions_by_location(expected_merged_data_and_predictions)
    metrics_vals = reporter.count_metrics([metrics.utils.get_metric('sklearn.mean_absolute_error')
                                           , metrics.utils.get_metric('MAPE')]
                                          , data_for_report)
    df3d = reporter.form_tables(metrics_vals
                                , splitting_features=['pred_type']
                                , row_features=['model', 'location']
                                , column_features=['metric', 'horizon'])
    assert expected_df3d.keys() == df3d.keys()
    for title, df2d in df3d.items():
        assert (expected_df3d[title].index.values == df2d.index.values).all()
        assert (expected_df3d[title].columns.values == df2d.columns.values).all()
        assert (expected_df3d[title].fillna(-1) == df2d.fillna(-1)).all().all()
