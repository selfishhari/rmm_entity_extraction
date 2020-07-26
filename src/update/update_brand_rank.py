import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import datetime
import sys
sys.path.append("/home/ec2-user/rmm_entity_extraction")

from src.extractor.fuzz_match import FuzzySearch
from conf.public import catalog, credentials


excel_sheet_loc = catalog.EXCEL_SHEET

dataframes =pd.ExcelFile(excel_sheet_loc)

telegram_question = catalog.TELEGRAM_QUESTION

telegram_question_df = pd.read_excel(dataframes, telegram_question)

output_path = catalog.CSV_OUTPUT_BRAND_RANK



class UpdateBrandRank():


    def __init__(self):
        self.telegram_question_df = telegram_question_df

        self.telegram_question_extract = self.telegram_question_df[['question ','brand ']]

        self.output_path = output_path

    def get_brand_count_rank(self):

        brand_split = pd.concat([Series(row['question '], str(row['brand ']).split('|'))
                         for _, row in self.telegram_question_extract.iterrows()]).reset_index()


        brand_split.rename(columns = {'index':'brand',0:'count'}, inplace = True)

        brand_split.brand = brand_split.brand.replace({"mi": "xiaomi",
                                                   "redmi": "xiaomi"})

        brand_counts = brand_split.groupby(['brand']).count()

        brand_counts = brand_counts.reset_index()

        brand_counts['brand_rank'] = brand_counts['count'].rank(ascending = 0,method='min')

        brand_counts = brand_counts.sort_values(by ='brand_rank' )

        brand_counts = brand_counts.reset_index()

        brand_counts['brand_rank'] = brand_counts['brand_rank'].astype(int)

        brand_counts = brand_counts[['brand','count','brand_rank']]

        return brand_counts

    def save_model(self,dataframe):

        # saving the model to a csv file
        dataframe.to_csv(self.output_path,index=False)

        return True

    def update(self):

        #extracting list of alternative names dict with brand as key
        brand_counts = self.get_brand_count_rank()

        #saving the dataframe
        self.save_model(brand_counts)

        return True

if __name__ == "__main__":
    brand_rank = UpdateBrandRank()
    brand_rank.update()