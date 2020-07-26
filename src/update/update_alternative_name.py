import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import datetime
import sys
sys.path.append("/home/ec2-user/rmm_entity_extraction")
from conf.public import catalog, credentials
from src.extractor.fuzz_match import FuzzySearch

excel_sheet_loc = catalog.EXCEL_SHEET

dataframes =pd.ExcelFile(excel_sheet_loc)

telegram_question = catalog.TELEGRAM_QUESTION

telegram_question_df = pd.read_excel(dataframes, telegram_question)

alternative_name_sheet = catalog.ALTERNATIVE_NAME

alternative_name = pd.read_excel(dataframes, alternative_name_sheet)

output_path = catalog.CSV_OUTPUT_ALTERNATIVE_NAME



class UpdateAlternativeName():


    def __init__(self):
        self.telegram_question_df = telegram_question_df

        self.alternative_name = alternative_name

        self.telegram_question_extract = self.telegram_question_df[['id','question ','brand ']]

        self.output_path = output_path

    def extract_alt_names(self):

        all_list = []

        #iterating over rows and finding alterante names based on brand annotated
        for index,row in self.telegram_question_extract.iterrows():

            words = (row['brand '].split("|"))

            string = (row['question '])

            spell_check = FuzzySearch(words)

            spell_check.check(string)

            all_list.extend(spell_check.suggestions())

        return all_list

    def list_to_dataframe(self,list_value):

        #chnaging list of dict into dataframe and dropping duplicates
        alt_df = pd.DataFrame(list_value)

        alt_df = alt_df.drop_duplicates()

        #grouping by common brand name and converting it into dataframe
        alt_df = alt_df.groupby('brand')['alt'].apply(list)

        alt_df = alt_df.to_frame()

        alt_df = alt_df.reset_index()

        return alt_df

    def merging_with_existing(self,alt_dataframe):

        # merging the extracted dataframe with existing alternative name sheet
        self.alternative_name['alt'] = self.alternative_name['alt'].apply(lambda x: x.split(','))

        alt_merged = alt_dataframe.merge(self.alternative_name,left_on='brand', right_on='brand')

        #concatinating matching alternate names
        alt_merged['alt'] = alt_merged['alt_x'] + alt_merged['alt_y']

        alt_merged = alt_merged [['brand','alt']]

        alt_merged['alt'] = alt_merged['alt'].apply(lambda x: ', '.join(map(str, x)))

        return alt_merged

    def concat_differences_new(self, df_1,df_2):

        #finding difference between merged datframe and extracted dataframe
        concated_model = pd.concat([df_1, df_2]).drop_duplicates(subset='brand',keep=False)

        concated_model['alt'] = concated_model['alt'].apply(lambda x: ', '.join(map(str, x)))

        #concatinating the difference with merged dataframe
        concated_model_df = pd.concat([df_2,concated_model])

        concated_model_df = concated_model_df.reset_index()

        # extracting required columns
        concated_model_df = concated_model_df[['brand','alt']]

        return concated_model_df

    def concat_differences_existing(self,df_1,df_2):

        #finding difference between merged datframe and old dataframe
        concated_model_2 = pd.concat([self.alternative_name, df_1]).drop_duplicates(subset='brand',keep=False)

        concated_model_2['alt'] = concated_model_2['alt'].apply(lambda x: ', '.join(map(str, x)))

        #concatinating the difference with merged dataframe
        concated_model_df = pd.concat([df_2,concated_model_2])

        concated_model_df = concated_model_df.reset_index()

        # extracting required columns
        concated_model_df = concated_model_df[['brand','alt']]

        return concated_model_df

    def save_model(self,dataframe):

        # saving the model to a csv file
        dataframe.to_csv(self.output_path,index=False)

        return True

    def update(self):

        #extracting list of alternative names dict with brand as key
        alt_list = self.extract_alt_names()

        # forming a dataframe out of it
        list_df = self.list_to_dataframe(alt_list)

        # merging the formed dataframe with existing alternative name dataframe
        merged_df = self.merging_with_existing(list_df)

        # concatinating the difference between merged and extracted dataframe
        concat_new = self.concat_differences_new(list_df,merged_df)

        # concatinating the difference between merged and old dataframes
        concat_old = self.concat_differences_existing(merged_df,concat_new)

        #saving the dataframe
        self.save_model(concat_old)

        return True

if __name__ == "__main__":
    update_alternative = UpdateAlternativeName()
    update_alternative.update()