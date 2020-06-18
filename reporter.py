'''Create a report form data and predictions'''

import warnings
import re
import datetime
from itertools import product
from collections import OrderedDict
from os.path import join as pathjoin, exists as pathexists
from urllib.request import urlretrieve

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

PANDAS_DATE_PATTERN = re.compile(r'(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})')
DATAFILE_PATTERN = re.compile(r'(?P<month>\d{2})-(?P<day>\d{2})-(?P<year>\d{4}).csv')
DATAFILE_TEMPLATE = '%m-%d-%Y.csv'
DATASOURCE = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/'\
             'csse_covid_19_data/csse_covid_19_daily_reports/'

class Reporter:
    def __init__(self, output_path_template, whole_country=False, datastorage_path='data'):
        self.opt = output_path_template
        self.whole_country = whole_country
        self.datastorage = datastorage_path

    def parse_file(self, filepath):
        def parse_date_obj(date_obj):
            if isinstance(date_obj, str):
                if PANDAS_DATE_PATTERN.match(date_obj):
                    date_template = '%Y-%m-%d'
                else:
                    slash_pos = date_obj.rfind('/')
                    if ((len(date_obj)-slash_pos == 3)
                            or (date_obj.find(' ')-slash_pos == 3)):
                        date_template = '%m/%d/%y'
                    else:
                        date_template = '%m/%d/%Y'
                space_index = date_obj.rfind(' ')
                if space_index > -1:
                    date_obj = date_obj[:space_index]
                return datetime.datetime.strptime(date_obj, date_template)
            result = []
            for date_str in date_obj:
                result.append(parse_date_obj(date_str))
            return result

        print(filepath)
        fileinfo = None
        file_pattern_res = DATAFILE_PATTERN.search(filepath)
        if file_pattern_res:
            fileinfo = pd.read_csv(filepath)
        try:
            fileinfo = pd.read_csv(filepath, parse_dates=['Date']
                                   , date_parser=parse_date_obj)
        except ValueError:
            try:
                fileinfo = pd.read_csv(filepath, parse_dates=['TrialDate']
                                       , date_parser=parse_date_obj)
            except ValueError:
                try:
                    fileinfo = pd.read_csv(filepath, parse_dates=['Last_Update']
                                           , date_parser=parse_date_obj)
                except ValueError:
                    try:
                        fileinfo = pd.read_csv(filepath, parse_dates=['Last Update']
                                               , date_parser=parse_date_obj)
                    except ValueError:
                        raise ValueError('In ' + filepath + ' .'
                                         ' Possible reasons (can be recognized by the exception stack):'
                                         ' 1) Date column is not recognized.'
                                         ' Please, use for the date column name'
                                         ' either "Date", "TrialDate", "Last_Update" or "Last Update"!'
                                         ' 2) Invalid data format.')
        fileinfo = fileinfo.rename({
            # Old data format
            'Province/State' : 'Region', 'Country/Region' : 'Country', 'Last Update' : 'Date',
            # New data format
            'Province_State' : 'Region', 'Country_Region' : 'Country', 'Last_Update' : 'Date',
            # Predictions
            'TrialDate': 'Date',
        }, axis='columns')

        if not(set(fileinfo.columns) >= set(['Region', 'Country', 'Date'])):
            fileinfo.columns = np.hstack([['Region', 'Country', 'Date']
                                          , fileinfo.columns.values[3:]])
        fileinfo[['Region', 'Country']] = fileinfo[['Region', 'Country']].fillna(value='')
        raw_index = ['Country', 'Region', 'Date']
        for col in ['Admin2', 'FIPS']:
            if col in fileinfo.columns:
                fileinfo[col] = fileinfo[col].fillna(value='')
                raw_index.append(col)

        if file_pattern_res:
            fileinfo['Date'] = pd.Timestamp('{}-{}-{}'.format(
                file_pattern_res.group('year')
                , file_pattern_res.group('month')
                , file_pattern_res.group('day')
            ))

        fileinfo = fileinfo.set_index(raw_index)

        if self.whole_country:
            fileinfo = fileinfo.groupby(level=['Country', 'Date']).sum(min_count=1)
            fileinfo.set_index(pd.MultiIndex.from_arrays(
                [
                    fileinfo.index.get_level_values('Country')
                    , [''] * fileinfo.shape[0]
                    , fileinfo.index.get_level_values('Date')
                ]
                , names=['Country', 'Region', 'Date']
            ), inplace=True)
        elif len(raw_index) > 3:
            fileinfo = fileinfo.groupby(level=['Country', 'Region', 'Date']).sum(min_count=1)

        fileinfo.sort_index(inplace=True)
        return fileinfo

    @staticmethod
    def get_locations_by_periods(country_region_date):
        start_dates = []
        end_dates = []
        countries = []
        regions = []
        cur_countries = set()
        cur_regions = set()
        last_countries = set()
        last_regions = set()
        last_date = None
        last_date_buf = None
        date = None
        for country, region, date in sorted(country_region_date, key=lambda x: x[2]):
            last_date = last_date_buf
            if last_date is None:
                start_dates.append(date)
            elif date == last_date:
                cur_countries.add(country)
                cur_regions.add(region)
                continue
            if (cur_countries != last_countries) or (cur_regions != last_regions):
                if len(last_countries) > 0:
                    end_dates.append(last_date)
                    start_dates.append(date)
                    countries.append(last_countries)
                    regions.append(last_regions)
                last_countries = cur_countries
                last_regions = cur_regions
            cur_countries = set([country])
            cur_regions = set([region])
            last_date_buf = date
        if (cur_countries != last_countries) or (cur_regions != last_regions):
            if len(last_countries) > 0:
                end_dates.append(last_date)
                start_dates.append(date)
                countries.append(last_countries)
                regions.append(last_regions)
        end_dates.append(date)
        countries.append(cur_countries)
        regions.append(cur_regions)
        assert ((len(start_dates) == len(end_dates))
                and (len(start_dates) == len(countries))
                and (len(start_dates) == len(regions))) \
               , ('Invalid (not equal) size (start_dates, end_dates, countries, regions): '
                  + ' '.join([len(start_dates), len(end_dates), len(countries), len(regions)]))
        return start_dates, end_dates, countries, regions

    def get_data_for_day(self, selected_date, countries=None, regions=None):
        datafile = selected_date.strftime(DATAFILE_TEMPLATE)
        datapath = pathjoin(self.datastorage, 'daily_reports', datafile)
        if not(pathexists(datapath)):
            urlretrieve(DATASOURCE+datafile, datapath)
        data = self.parse_file(datapath)
        data = data[['Confirmed', 'Deaths', 'Recovered']]

        if not(countries is None):
            data = data[data.index.get_level_values('Country').isin(countries)]
        if not(regions is None):
            regions.add('')
            data = data[data.index.get_level_values('Region').isin(regions)]
        return data

    def get_data_for_period(self, start_date, end_date, countries=None, regions=None):
        if isinstance(start_date, list):
            if isinstance(end_date, list):
                return pd.concat([self.get_data_for_period(sd, ed, countries=c, regions=r)
                                  for sd, ed, c, r in zip(start_date, end_date, countries, regions)])
            return pd.concat([self.get_data_for_period(sd, end_date, countries=c, regions=r)
                              for sd, c, r in zip(start_date, countries, regions)])
        if isinstance(start_date, list):
            return pd.concat([self.get_data_for_period(start_date, ed, countries=c, regions=r)
                              for ed, c, r in zip(end_date, countries, regions)])
        datafile = (('' if countries is None else ('+'.join(sorted(countries)) + '_'))
                    + ('' if regions is None else ('+'.join(sorted(regions)) + '_'))
                    + start_date + '_'
                    + end_date + '.csv')
        datapath = pathjoin(self.datastorage, 'time_series', datafile)
        if pathexists(datapath):
            return self.parse_file(datapath)
        data_by_days = []
        for current_date in pd.date_range(start=start_date, end=end_date):
            data_by_days.append(self.get_data_for_day(current_date
                                                      , countries=countries, regions=regions))
        data = pd.concat(data_by_days)
        data.to_csv(datapath)
        return data

    @staticmethod
    def trial_date_to_prediction_date(prediction):
        max_gap = max(map(lambda x: int(x.split('_')[1]), prediction.columns))

        new_multiindex = []
        for country, country_data in prediction.groupby(level=0):
            for region, country_region_data in country_data.groupby(level=1):
                dates = country_region_data.index.get_level_values(2)
                min_date = dates.min()
                max_date = dates.max() + pd.Timedelta(max_gap, unit='d')
                dates = pd.date_range(start=min_date, end=max_date)
                new_multiindex += product([country], [region], dates)
        prediction = prediction.reindex(new_multiindex)
        for column in prediction.columns:
            gap = int(column.split('_')[1])
            prediction[column] = np.roll(prediction[column], gap)
        prediction = prediction.dropna(how='all')
        return prediction


    @classmethod
    def prepare_predictions_for_merge(cls, predictions):
        for predictor_name, prediction in predictions.items():
            prediction.columns = [col + '_' + predictor_name for col in prediction.columns]
        predictions_val = list(predictions.values())
        merged_pr = predictions_val[0].join(predictions_val[1:], how='outer')
        return cls.trial_date_to_prediction_date(merged_pr)

    def merge_data_and_predictions(self, data, merged_predictions):
        return data.join(merged_predictions, how='right')

    def group_merged_data_and_predictions_by_location(self, merged_data_and_predictions):
        grouped = {}
        for country, country_data in merged_data_and_predictions.groupby(level=0):
            for region, country_region_data in country_data.groupby(level=1):
                location = country
                if len(region.strip()) > 0:
                    location += '/' + region
                grouped[location] = country_region_data
        return grouped

    def prepare_data_and_prediction_for_report(self, predictions_list):
        predictions = {prediction_file[max(prediction_file.rfind('/'), prediction_file.rfind('\\')) + 1
                                       : prediction_file.rfind('.')]
                       : self.parse_file(prediction_file)
                       for prediction_file in predictions_list}
        merged_predictions = self.prepare_predictions_for_merge(predictions)
        predictions_index = merged_predictions.index.to_native_types()
        start_dates, end_dates, countries, regions = self.get_locations_by_periods(predictions_index)
        # print('\n'.join(map(str, zip(start_dates, end_dates, countries, regions))))
        data = self.get_data_for_period(start_dates, end_dates, countries=countries, regions=regions)
        merged = self.merge_data_and_predictions(data, merged_predictions).sort_index()
        return merged

    @staticmethod
    def count_metrics(metrics_list, data_for_report, horizons=None):
        metric_vals = {
            'pred_type': [],
            'location': [],
            'metric': [],
            'horizon': [],
            'model': [],
            'value': [],
        }
        for location, dap in data_for_report.items():

            for column in dap.columns[3:]:
                pred_type, horizon, model_name = column.split('_')
                if not((horizons is None) or (horizon in horizons)):
                    continue
                horizon = int(horizon)
                mask = dap[pred_type].notna() & dap[column].notna()
                if (~mask).all():
                    continue
                for metric in metrics_list:
                    metric_vals['pred_type'].append(pred_type)
                    metric_vals['location'].append(location)
                    metric_vals['metric'].append(metric.metric_name)
                    metric_vals['horizon'].append(horizon)
                    metric_vals['model'].append(model_name)
                    metric_vals['value'].append(metric(dap[pred_type][mask].values
                                                       , dap[column][mask].values))
        return metric_vals

    @staticmethod
    def form_tables(metric_vals, splitting_features, row_features, column_features):
        def make_label(selector, values, i, as_tuple=False):
            if len(selector) == 1:
                return values[selector[0]][i]
            label = tuple([values[s][i] for s in selector])
            if as_tuple:
                return label
            return '_'.join(label)

        #def make_labels(selector, values):
        #    if len(selector) == 1:
        #        return list(set(values[selector[0]]))
        #    return ['_'.join(el) for el in set(zip(*[values[s] for s in selector]))]

        # splitting_labels = make_labels(splitting_features, metric_vals)
        # row_labels = make_labels(row_features, metric_vals)
        # column_labels = make_labels(column_features, metric_vals)
        #         # tables = {feature : pd.DataFrame(index=row_labels, columns=column_labels)
        #           for feature in splitting_labels}

        table3d = {}
        for i in range(len(metric_vals['value'])):
            splitting_label = make_label(splitting_features, metric_vals, i)
            row_label = make_label(row_features, metric_vals, i, as_tuple=True)
            column_label = make_label(column_features, metric_vals, i, as_tuple=True)
            if not(splitting_label in table3d):
                table3d[splitting_label] = {}
            if not(column_label in table3d[splitting_label]):
                table3d[splitting_label][column_label] = []
            table3d[splitting_label][column_label].append((row_label, metric_vals['value'][i]))

        get_element0 = lambda x: x[0]
        df3d = OrderedDict(
            [(spl, pd.DataFrame.from_dict(
                OrderedDict([(cl, OrderedDict(sorted(row, key=get_element0)))
                             for cl, row in sorted(table.items(), key=get_element0)]))
             ) for spl, table in sorted(table3d.items(), key=get_element0)]
        )

        return df3d

    @staticmethod
    def df2tex(dataf):
        linet = ' \\\\\n'
        text = dataf.round({
            'MAE_Germany': 1,
            'MAE_Russia': 1,
            'MAE_US': 1,
            'MALE_Germany': 3,
            'MALE_Russia': 3,
            'MALE_US': 3,
            'MALE_Germany': 3,
            'MALE_Russia': 3,
            'MALE_US': 3,
            'MASE_Germany': 3,
            'MASE_Russia': 3,
            'MASE_US': 3,
        }).to_csv(sep='&', line_terminator=linet)[:-len(linet)]
        return text.replace('&', ' & ')

    @staticmethod
    def generate_forecasting_plot(data_for_report, save_path_template
                                  , date_selector=None
                                  , axis_x=None, axis_y_splitter=None
                                  , line_labels_basis=None):
        if date_selector is None or len(date_selector) == 0:
            return []
        if axis_x is None:
            axis_x = 'Horizon'
        if axis_y_splitter is None:
            axis_y_splitter = 'PredType'
        if line_labels_basis is None:
            line_labels_basis = ['Model']

        generated_files = []

        warnings.warn("Generating forecasting plots are not implemented now"
                      , category=FutureWarning)
        return generated_files

    @staticmethod
    def generate_compairing_plot(metric_vals, save_path_template
                                 , axis_x=None, axis_y=None
                                 , line_labels_basis=None):
        if axis_x is None:
            axis_x = 'Date'
        if axis_y is None:
            axis_y = ['Confirmed', 'Deaths', 'Recovered']
        if line_labels_basis is None:
            line_labels_basis = ['Model', 'Horizon']

        generated_files = []

        warnings.warn("Generating compairing plots are not implemented now"
                      , category=FutureWarning)
        return generated_files

    def report(self, predictions_list, metrics_list, horizons_list=None, date_selector=None):
        print('Preparing data for report')
        merged_data_and_predictions = self.prepare_data_and_prediction_for_report(predictions_list)
        data_for_report = self.group_merged_data_and_predictions_by_location(merged_data_and_predictions)
        print('Counting metrics')
        metric_vals = self.count_metrics(metrics_list, data_for_report, horizons_list)
        print('Forming tables')
        df3d = self.form_tables(metric_vals
                                , splitting_features=['pred_type']
                                , row_features=['horizon', 'model']
                                , column_features=['metric', 'location'])

        print('Generating output')
        generated_files = []
        merged_data_and_predictions.to_csv(self.opt + '_data.csv')
        generated_files.append(self.opt + '_data.csv')

        for title, df2d in df3d.items():
            df2d.to_csv(self.opt + '_metrics_' + title + '.csv')
            generated_files.append(self.opt + '_metrics_' + title + '.csv')

            # multi2single
            df2d.index = df2d.index.map(lambda x: '_'.join(map(str, x)))
            df2d.columns = df2d.columns.map(lambda x: '_'.join([str(y) for y in x]))

            with open(self.opt + '_metrics_' + title + '.tex', mode='w') as texfile:
                texfile.write(self.df2tex(df2d))
            generated_files.append(self.opt + '_metrics_' + title + '.tex')

            generated_files += self.generate_forecasting_plot(data_for_report, self.opt + '_metrics_' + title
                                                              , date_selector=date_selector)
            generated_files += self.generate_compairing_plot(metric_vals, self.opt + '_metrics_' + title)

        return generated_files
