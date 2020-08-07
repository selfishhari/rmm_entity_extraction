import requests
import pandas as pd
import json

from conf.public import catalog


populate_set = catalog.POPULATE_SET

populate_data = pd.read_csv(populate_set)


class PopulateData:
    def __init__(self):

        self.dataframe = populate_data

        self.url = 'http://0.0.0.0:5000/extract'


    def data_formulation(self):
        all_data =[]

        for index,row in self.dataframe.iterrows():

            data = {}

            data['text']=row['messages']

            data['tag']= row['tag']

            data['message_id']=row['message_id']

            data['id']=row['user_id']

            data['date']=row['date']

            all_data.append((data))

        return all_data

    def populate_data(self,all_data):
        all_response = []

        for data in all_data:

            response = requests.post(self.url, json=data)

            print(response.text)
            all_response.append(response.text)

        return all_response

    def populate(self):
        all_data = self.data_formulation()

        responses = self.populate_data(all_data)


if __name__ == "__main__":

    populate_data = PopulateData()
    responses = populate_data.populate()
