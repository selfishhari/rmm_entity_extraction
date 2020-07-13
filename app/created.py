from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import json
import pandas as pd
import numpy as np
from datetime import date
import string
import sqlalchemy
from conf.public import catalog, credentials


excel_sheet_loc = catalog.EXCEL_SHEET
dataframes =pd.ExcelFile(excel_sheet_loc)


brand_model_sheet = catalog.BRAND_MODEL
alternative_name_sheet = catalog.ALTERNATIVE_NAME

brand_model = pd.read_excel(dataframes, brand_model_sheet)
alternative_name = pd.read_excel(dataframes,alternative_name_sheet)


host = catalog.DB_HOST
user_name = credentials.DB_USERNAME
password = credentials.DB_PASSWORD
db_name = catalog.DB_NAME

def _is_strict_alnum(x):

    if str(x).isdigit():
        return False

    o_len = len(str(x))

    remove_digits = str.maketrans('', '', string.digits)

    procx = str(x).translate(remove_digits)

    p_len = len(procx)

    return o_len > p_len

class MatchUser():
    def __init__(self,input_object):
        print(input_object)
        self.alternative_name = alternative_name
        self.brand_model = brand_model
        self.input_object = input_object
        self.tag = self.input_object['tag']
        self.user_id = self.input_object['id']
        self.text = self.input_object['text']
        self.db_engine = sqlalchemy.create_engine(
            'mysql+pymysql://{0}:{1}@{2}/{3}?charset=utf8mb4'.format(user_name,
                                             password,
                                             host,
                                             db_name))


    def pre_processing(self):
        string_1 = self.text
        string_1 = re.sub("(?P<url>https?://[^\s]+)","", string_1)
        string_1 = string_1.lower()
        string_1= string_1.replace("\n", " ")
        string_1 = string_1.translate ({ord(c): " " for c in "!@#$%^&*()[]{};:,./<>?\|`~=_"})
        string_1 = " ".join(string_1.split())
        printable = set(string.printable)
        string_1 = ''.join(filter(lambda x: x in printable, string_1))
        return string_1

    def stop_words_removal(self,pre_processed_text):
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(pre_processed_text)
        filtered_tokens = [w for w in word_tokens if not w in stop_words]
        return filtered_tokens

    def _search_brands(self,items_list, x_list, exact_search=True):
        items = []
        for item in items_list:
            if (exact_search == True) | (len(item)<4):
                for w in x_list:
                    if item == w.strip():
                        items += [item]
                        #break;
            else:
                if item in x:
                    items += [item]
        alt_names = self._search_alt_brand_name(x_list)
        items.extend(alt_names)
        items = list(set(items)) #removing duplicates
        if "" in items:
            items.remove("")
            if items == None:
                items = []
        return items


    def _search_alt_brand_name(self,x_list):
        alternative_name_copy = self.alternative_name.copy()   #-------conf
        alternative_name_copy['alt'] = alternative_name_copy['alt'].apply(lambda x: str(x).split(","))
        alt_name = alternative_name_copy.set_index('brand')['alt'].to_dict()
        brand_name = []
        for item in x_list:
            for key, value in alt_name.items():
                if item.strip() in value:
                    brand_name.append(key)
        return brand_name

    def check(self,string, sub_str):
        if (string.find(sub_str) == -1):
            return False
        else:
            return True

    def _search_models_based_on_brand(self,string,brands):
        print(string)
        models = []
        for brand in brands:
            sub_set_models = self.brand_model.loc[brand_model['brand'] == brand, 'models'].str.lower().unique().tolist()
            for model in sub_set_models:
                if self.check(string,str(model)):
                    string = string.replace(str(model),'')
                    models.append({'brand':brand,'model':model})
            if any(brand in model for model in models):
                continue
            else:
                models.append({'brand':brand,'model':None})

        return models, string

    def _search_models(self,string,rem_num_only = True):
        print(string)
        all_models = self.brand_model['models'].str.lower().unique().tolist()
        models= []

        if rem_num_only:
            for item in all_models:
                if str(item).isdigit():
                    all_models.remove(item)

        x_list = string.split(" ")
        for model in all_models:
            if (len(str(model)) < 2) | (not _is_strict_alnum(model)) | (len(str(model).split(" ")) < 2):
                for w in x_list:
                    if w == model:
                        brands = self.brand_model.loc[brand_model['models'] == model, 'brand']
                        for brand in brands:
                            models.append({'brand':brand,'model':model})

            else:
                if self.check(string,str(model)):
                    brands = self.brand_model.loc[brand_model['models'] == model, 'brand']
                    for brand in brands:
                        models.append({'brand':brand,'model':model})

        return models

    def extract_models(self,string,brands):
        all_models = []
        extracted_models, altered_string = (self._search_models_based_on_brand(string,brands))
        all_models.extend(extracted_models)
        all_models.extend((self._search_models(altered_string)))
        extracted_dataframe = pd.DataFrame(all_models)
        extracted_dataframe['user_id']= self.user_id
        extracted_dataframe['tag']= self.tag

        if 'date' in self.input_object:
            trans_date = self.input_object['date']
        else:
            trans_date = date.today()

        extracted_dataframe['date']= str(trans_date)
        extracted_dataframe = extracted_dataframe[['user_id','date','tag','brand','model']]
        extracted_dataframe = extracted_dataframe.dropna()
        return extracted_dataframe

    def extract_brand_model(self):
        preprocessed_text = self.pre_processing()
        tokens = self.stop_words_removal(preprocessed_text)
        all_brands = self.brand_model['brand'].str.lower().unique().tolist()
        brands_list = self._search_brands(all_brands,tokens)
        extracted_dataframe = self.extract_models(preprocessed_text,brands_list)
        return extracted_dataframe

    def get_pincode(self):
        pin_code_query = "SELECT pin FROM shops WHERE user_id = {};".format(self.user_id)
        pin_code_result = self.db_engine.execute(pin_code_query)
        pin_code_list = [r[0] for r in pin_code_result]
        return pin_code_list[0]

    def generate_pincodes(self,pincode,num=5):
        pincodes = []
        for i in range(pincode-num,pincode):
            pincodes.append(i)

        for j in range(pincode,pincode+num):
            pincodes.append(j)

        return tuple(pincodes)

    def get_matching_user(self,pincodes):
        user_id_query = "SELECT user_id FROM shops WHERE pin IN {};".format(pincodes)
        user_id_result = self.db_engine.execute(user_id_query)
        user_id_list = [r[0] for r in user_id_result]
        return user_id_list

    def _dataframe_available(self,user_id_list):
        if self.tag =='sell':
            tag = 'buy'
        elif self.tag =='buy':
            tag = 'sell'
        if len(user_id_list)>1:
            extracted_model_query = "SELECT * from extracted WHERE user_id IN {} AND tag = '{}';".format(tuple(user_id_list),tag)
        elif len(user_id_list)==1:
            extracted_model_query = "SELECT * from extracted WHERE user_id = {} AND tag = '{}';".format(user_id_list[0],tag)
        available_models = pd.read_sql_query(extracted_model_query,self.db_engine)
        return available_models

    def get_existing_df(self):
        existing_df_query = "SELECT * FROM {};".format("extracted") #----table names in catalog
        existing_df = pd.read_sql_query(existing_df_query,self.db_engine)
        return existing_df

    def uploaded_dateframe(self,dataframe):
        try:
            dataframe.to_sql('extracted', con=self.db_engine, if_exists='replace',index=False)
            return True
        except Exception as e:
            print(str(e))
            return False

    def get_users(self,extracted_dataframe,available_models):

        final_df = pd.merge(extracted_dataframe, available_models, on=['brand','model'], how='inner')

        if final_df.empty:

            final_df = pd.merge(extracted_dataframe, available_models, on=['brand'], how='inner')

            if final_df.empty:

                final_df = available_models

        final_list = final_df['user_id'].tolist()

        return final_list

    def matched_users(self):
        print("1")
        extracted_dataframe = self.extract_brand_model()
        existing_df = self.get_existing_df()

        if existing_df.empty:
            upload_df_final = extracted_dataframe
        else:
            upload_df = pd.concat([existing_df,extracted_dataframe])
            upload_df_final = upload_df.drop_duplicates()

        upload_status = self.uploaded_dateframe(upload_df_final)
        if upload_status:
            available_dataframe = pd.DataFrame()
            num = 5

            while(available_dataframe.empty):
                print(num)
                pincode = self.get_pincode()
                pincodes_tuple = self.generate_pincodes(pincode,num)
                matching_users = self.get_matching_user(pincodes_tuple)
                print(matching_users)
                matching_users = list(set(matching_users))
                print(matching_users)
                available_dataframe = self._dataframe_available(matching_users)
                num += 1


            extracted_brand_model_df = extracted_dataframe[['brand','model']]
            matched_users = self.get_users(extracted_brand_model_df,available_dataframe)
            user_dict = {}
            for i in range(len(set(matched_users))):
                user_dict[i] = matched_users[i]

            return json.dumps(user_dict)
