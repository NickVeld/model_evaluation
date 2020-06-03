'''Create a report form data and predictions'''

from itertools import chain
from os.path import join as pathjoin, exists as pathexists
from urllib.request import urlretrieve

import pandas as pd

DATAFILE_TEMPLATE = '%m-%d-%Y.csv'
DATASOURCE = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/'

class Reporter:
    def __init__(self, output_path_template, datastorage_path='data'):
        self.opt = output_path_template
        self.datastorage = datastorage_path

    @staticmethod
    def parse_file(filepath):    
        fileinfo = pd.read_csv(filepath)
        fileinfo.columns = (['Region', 'Country', 'TrialDate']
                            + chain.from_iterable([['Infected_' + i, 'Died_' + i, 'Recovered_' + i]
                                                   for i in range((len(fileinfo.columns)-3) // 3)]))
        return fileinfo

    def get_data_for_day(self, selected_date, countries=None, regions=None):
        datafile = selected_date.strftime(DATAFILE_TEMPLATE)
        if not(pathexists(pathjoin(self.datastorage, datafile))):
            urlretrieve(DATASOURCE+datafile, datafile)
        data = self.parse_file(datafile)
        
        if not(countries is None):
            data = data[data['Country'].isin(countries)]
        if not(regions is None):
            regions.add('')
            data = data[data['Region'].isin(regions)]
        return data
    
    def group_merged_data_and_predictions(self, data, predictons):
        pass

    def report(self, predictions_list):
        pass