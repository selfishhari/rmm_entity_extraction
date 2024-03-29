import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import datetime
import sys
sys.path.append("/home/ec2-user/rmm_entity_extraction")

from conf.public import catalog, credentials

excel_sheet_loc = catalog.EXCEL_SHEET

dataframes =pd.ExcelFile(excel_sheet_loc)

telegram_question = catalog.TELEGRAM_QUESTION

telegram_question_df = pd.read_excel(dataframes, telegram_question)

brand_model_sheet = catalog.BRAND_MODEL

brand_model = pd.read_excel(dataframes, brand_model_sheet)

output_path = catalog.CSV_OUTPUT_BRAND_MODEL



class UpdateBrandModel():


    def __init__(self):
        self.telegram_question_df = telegram_question_df

        self.brand_model = brand_model

        self.telegram_question_extract = self.telegram_question_df[['brand ','model ','id']]

        self.output_path = output_path

    def expand_brand_models(self):

        #Iterating over rows of dataframe
        for index, row in self.telegram_question_extract.iterrows():

            if type(row['model ']) == datetime.datetime:

                # dropping rows of type-datetime
                self.telegram_question_extract.drop(index, inplace=True)



        for index, row in self.telegram_question_extract.iterrows():

            if row['brand '].find('|') != -1:
                #dropping rows with multiple brands
                self.telegram_question_extract.drop(index, inplace=True)

        #splitting models at '|'
        model_split = pd.concat([Series(row['brand '], str(row['model ']).split('|'))
                         for _, row in self.telegram_question_extract.iterrows()]).reset_index()

        model_split.rename(columns = {'index':'models',0:'brand'}, inplace = True)

        #dropping duplicates and nan values
        model_split = model_split.drop_duplicates()

        model_split.dropna()

        return model_split

    def concat_brands_models(self, extracted):

        #concatting both dataframes and removing duplicates and nan values
        all_models = pd.concat([extracted,self.brand_model])

        all_models = all_models.drop_duplicates()

        all_models = all_models.dropna()

        return all_models

    def check(self,string, sub_str):

        if (str(string).find(str(sub_str)) == -1):

            return False

        else:

            return True

    def save_model(self,dataframe):

        dataframe.to_csv(self.output_path,index=False)

        return True

    def update(self):
        #extracting model from telegram questions
        brand_model = self.expand_brand_models()

        #concating dataframe with existing dataframe
        all_models = self.concat_brands_models(brand_model)

        #saving models
        save_model = self.save_model(all_models)

        return True


if __name__ == "__main__":
    update_brand = UpdateBrandModel()
    update_brand.update()