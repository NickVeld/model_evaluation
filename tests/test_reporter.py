import pandas as pd
import numpy as np

from reporter import Reporter

def test_constructor():
    Reporter(output_path_template="output/test")

def test_get_locations_by_periods():
    reporter = Reporter(output_path_template="output/test")
    prediction_index = np.array([('Japan', '', '2020-01-23'), ('Japan', '', '2020-01-27'),
                                 ('Mainland China', 'Beijing', '2020-01-23')], dtype=object)

    start_dates, end_dates, countries, regions = reporter.get_locations_by_periods(prediction_index)

    assert start_dates == ['2020-01-23', '2020-01-27']
    assert end_dates == ['2020-01-23', '2020-01-27']
    assert countries == [{'Mainland China', 'Japan'}, {'Japan'}]
    assert regions == [{'', 'Beijing'}, {''}]

def test_prepare_data_and_prediction_for_report():
    reporter = Reporter(output_path_template="output/test")
    prediction_name = 'TestPrediction'

    merged_data_and_predictions = reporter.prepare_data_and_prediction_for_report([
        'predictions/' + prediction_name + '.csv'
    ])

    expected = pd.DataFrame({
        'Infected': [1, 4, 22],
        'Dead': [np.nan, np.nan, np.nan],
        'Recovered': [np.nan, 1, np.nan],
        'Infected_1_TestPrediction': [5, np.nan, 10],
        'Dead_1_TestPrediction': [1, np.nan, 1],
        'Recovered_1_TestPrediction': [np.nan, np.nan, 3],
        'Infected_5_TestPrediction': [np.nan, 3, np.nan],
        'Dead_5_TestPrediction': [np.nan, 4, np.nan],
        'Recovered_5_TestPrediction': [np.nan, np.nan, np.nan],
    }, index=pd.MultiIndex.from_tuples([
        ('Japan', '', pd.Timestamp('20200123')),
        ('Japan', '', pd.Timestamp('20200127')),
        ('Mainland China', 'Beijing', pd.Timestamp('20200123')),
        #('Mainland China', 'Beijing', pd.Timestamp('20200127')),
    ], names=('Country', 'Region', 'Date'))).sort_index().astype(np.float64)

    #print(merged_data_and_predictions.index.values, expected.index.values)
    #print(merged_data_and_predictions.columns.values, expected.columns.values)
    #print(merged_data_and_predictions)
    #print(expected)
    assert (merged_data_and_predictions.index.values == expected.index.values).all()
    assert (merged_data_and_predictions.columns.values == expected.columns.values).all()
    #print((merged_data_and_predictions == expected).all())
    assert (merged_data_and_predictions.fillna(-1) == expected.fillna(-1)).all().all()
