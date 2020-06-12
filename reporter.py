'''Create a report form data and predictions'''

from itertools import product  # , chain
from os.path import join as pathjoin, exists as pathexists
from urllib.request import urlretrieve
import re
import datetime

import pandas as pd
import numpy as np

PANDAS_DATE_PATTERN = re.compile(r'(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})')
DATAFILE_TEMPLATE = '%m-%d-%Y.csv'
DATASOURCE = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/'\
             'csse_covid_19_data/csse_covid_19_daily_reports/'

class Reporter:
    def __init__(self, output_path_template, datastorage_path='data'):
        self.opt = output_path_template
        self.datastorage = datastorage_path

    @staticmethod
    def parse_file(filepath):
        def parse_date_obj(date_obj):
            #print(date_obj)
            #print(type(date_obj))
            if isinstance(date_obj, str):
                if PANDAS_DATE_PATTERN.match(date_obj):
                    date_template = '%Y-%m-%d'
                else:
                    date_template = '%m/%d/%y'
                space_index = date_obj.rfind(' ')
                if space_index > -1:
                    date_obj = date_obj[:space_index]
                return datetime.datetime.strptime(date_obj, date_template)
            #print('c', date_obj.shape)
            result = []
            for date_str in date_obj:
                result.append(parse_date_obj(date_str))
            return result

        #print(filepath)
        fileinfo = pd.read_csv(filepath, parse_dates=[2], date_parser=parse_date_obj)
        if set(fileinfo.columns.values[:3]) != set(['Region', 'Country', 'Date']):
            fileinfo.columns = np.hstack([['Region', 'Country', 'Date']
                                          , fileinfo.columns.values[3:]])
        #print(fileinfo['Date'])
        fileinfo[['Region', 'Country']] = fileinfo[['Region', 'Country']].fillna(value='')
        return fileinfo.set_index(['Country', 'Region', 'Date']).sort_index()

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
        for country, region, date in sorted(country_region_date, key=lambda x: x[2]):
            if last_date is None:
                start_dates.append(date)
                last_date = date
            elif date == last_date:
                cur_countries.add(country)
                cur_regions.add(region)
                continue
            if (cur_countries != last_countries) or (cur_regions != last_regions):
                end_dates.append(last_date)
                start_dates.append(date)
                countries.append(cur_countries)
                regions.append(cur_regions)
                last_countries = cur_countries
                last_regions = cur_regions
                last_date = date
            cur_countries = set([country])
            cur_regions = set([region])
        end_dates.append(last_date)
        countries.append(cur_countries)
        regions.append(cur_regions)
        assert ((len(start_dates) == len(end_dates))
                and (len(start_dates) == len(countries))
                and (len(start_dates) == len(regions))) \
               , ("Invalid (not equal) size (start_dates, end_dates, countries, regions): "
                  + " ".join([len(start_dates), len(end_dates), len(countries), len(regions)]))
        return start_dates, end_dates, countries, regions

    def get_data_for_day(self, selected_date, countries=None, regions=None):
        datafile = selected_date.strftime(DATAFILE_TEMPLATE)
        if not(pathexists(pathjoin(self.datastorage, datafile))):
            urlretrieve(DATASOURCE+datafile, pathjoin(self.datastorage, datafile))
        data = self.parse_file(pathjoin(self.datastorage, datafile))
        data.columns = ["Infected", "Dead", "Recovered"]

        if not(countries is None):
            data = data[data.index.get_level_values('Country').isin(countries)]
        if not(regions is None):
            regions.add('')
            data = data[data.index.get_level_values('Region').isin(regions)]
        return data

    def get_data_for_period(self, start_date, end_date, countries=None, regions=None):
        # print('\n', countries, regions, start_date, end_date)
        if isinstance(start_date, list):
            if isinstance(end_date, list):
                return pd.concat([self.get_data_for_period(sd, ed, countries=c, regions=r)
                                  for sd, ed, c, r in zip(start_date, end_date, countries, regions)])
            return pd.concat([self.get_data_for_period(sd, end_date, countries=c, regions=r)
                              for sd, c, r in zip(start_date, countries, regions)])
        elif isinstance(start_date, list):
            return pd.concat([self.get_data_for_period(start_date, ed, countries=c, regions=r)
                              for ed, c, r in zip(end_date, countries, regions)])
        datafile = (('' if countries is None else ('+'.join(sorted(countries)) + '_'))
                    + ('' if regions is None else ('+'.join(sorted(regions)) + '_'))
                    + start_date + '_'
                    + end_date + '.csv')
        datapath = pathjoin(self.datastorage, datafile)
        if pathexists(datapath):
            return self.parse_file(datapath)
        data_by_days = []
        # print(start_date, end_date, pd.date_range(start=start_date, end=end_date))
        for current_date in pd.date_range(start=start_date, end=end_date):
            #print(current_date)
            data_by_days.append(self.get_data_for_day(current_date
                                                      , countries=countries, regions=regions))
        data = pd.concat(data_by_days)
        data.to_csv(datapath)
        return data

    @staticmethod
    def trial_date_to_prediction_date(prediction):
        max_gap = max(map(lambda x: int(x.split('_')[1]), prediction.columns))
        #prediction.index.names = ['Country', 'Region', 'Date']
        new_multiindex = []
        for country, country_data in prediction.groupby(level=0):
            for region, country_region_data in country_data.groupby(level=1):
                dates = country_region_data.index.get_level_values(2)
                min_date = dates.min()
                max_date = dates.max() + pd.Timedelta(max_gap, unit='d')
                dates = pd.date_range(start=min_date, end=max_date)
                new_multiindex += product([country], [region], dates)
        prediction = prediction.reindex(new_multiindex)
        # print(prediction)
        for column in prediction.columns:
            gap = int(column.split('_')[1])
            prediction[column] = np.roll(prediction[column], gap)
        prediction = prediction.dropna(how='all')
        # print(prediction)
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
                grouped[country + '/' + region] = country_region_data
        return grouped

    def prepare_data_and_prediction_for_report(self, predictions_list):
        predictions = {prediction_file[prediction_file.rfind('/') + 1 : prediction_file.rfind('.')]
                       : self.parse_file(prediction_file)
                       for prediction_file in predictions_list}
        merged_predictions = self.prepare_predictions_for_merge(predictions)
        predictions_index = merged_predictions.index.to_native_types()
        start_dates, end_dates, countries, regions = self.get_locations_by_periods(predictions_index)
        data = self.get_data_for_period(start_dates, end_dates, countries=countries, regions=regions)
        #print('\n', data.sort_index().index.values, '\n', merged_predictions.index.values)
        merged = self.merge_data_and_predictions(data, merged_predictions).sort_index()
        return merged

    def report(self, predictions_list):
        merged_data_and_predictions = self.prepare_data_and_prediction_for_report(predictions_list)
        data_for_report = self.group_merged_data_and_predictions_by_location(merged_data_and_predictions)
        merged_data_and_predictions.to_csv(self.opt + ".csv")
        