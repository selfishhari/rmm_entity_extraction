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

#brand_model = pd.read_excel(dataframes, brand_model_sheet)
brand_model = pd.read_csv(catalog.brand_model_csv)

alternative_name = pd.read_excel(dataframes,alternative_name_sheet)


host = catalog.DB_HOST

user_name = credentials.DB_USERNAME

password = credentials.DB_PASSWORD

db_name = catalog.DB_NAME

ce_table = catalog.CUSTOMER_ENTITY_TABLE

shop_table = catalog.SHOP_TABLE

date_column = catalog.DATE_COLUMN

interval = catalog.INTERVAL

iterate_limit = catalog.ITERATE_LIMIT


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

        self.alternative_name = alternative_name

        self.brand_model = brand_model

        self.input_object = input_object

        self.shop_table = shop_table

        self.ce_table = ce_table

        self.INTERVAL = interval

        self.iterate_limit = iterate_limit

        self.DATE_COLUMN = date_column

        self.tag = self.input_object['tag']

        self.user_id = self.input_object['id']

        self.text = self.input_object['text']

        self.message_id = self.input_object['message_id']

        self.db_engine = sqlalchemy.create_engine(
            'mysql+pymysql://{0}:{1}@{2}/{3}?charset=utf8mb4'.format(user_name,
                                             password,
                                             host,
                                             db_name))


    def pre_processing(self):
        """
        Removes links, blanks, None, lowercasing, replacinh newline chars.

        Returns: string
            1. Preprocessed string
        """
        string_1 = self.text

        string_1 = re.sub("(?P<url>https?://[^\s]+)","", string_1)

        string_1 = string_1.lower()

        string_1= string_1.replace("\n", " ")

        string_1 = string_1.translate ({ord(c): " " for c in "!@#$%^&*()[]{};:,./<>?\|`~=_"})

        string_1 = " ".join(string_1.split())

        printable = set(string.printable)

        string_1 = ''.join(filter(lambda x: x in printable, string_1))

        return string_1

    def _stop_words_removal(self,pre_processed_text):
        """To remove stop words from the text

        Returns: List
            1. Filtered Tokens
        """

        stop_words = set(stopwords.words('english'))

        word_tokens = word_tokenize(pre_processed_text)

        filtered_tokens = [w for w in word_tokens if not w in stop_words]

        return filtered_tokens

    def _search_brands(self,items_list, x_list, exact_search=True):
        """To search brands from excel sheet matching with the text

        Returns: List
            1. Brand List
        """

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
        """To search alternative names of brand from excel sheet

        Returns: List
            1. Alternative brands
        """

        alternative_name_copy = self.alternative_name.copy()   #-------conf

        alternative_name_copy['alt'] = alternative_name_copy['alt'].apply(lambda x: str(x).split(","))

        alt_name = alternative_name_copy.set_index('brand')['alt'].to_dict()

        brand_name = []

        for item in x_list:

            for key, value in alt_name.items():

                if item.strip() in value:

                    brand_name.append(key)

        return brand_name

    def _check(self,string, sub_str):
        """To check if a sub string is present inside a string

        Returns: Boolean
        """

        if (string.find(sub_str) == -1):

            return False

        else:

            return True

    def _search_models_based_on_brand(self,string,brands):
        """To search models based on string and brands extracted

        Returns: Dataframe, string
            1. Model List
            2. Altered String
        """

        models = []

        x_list = string.split(" ")

        for brand in brands:

            sub_set_models = self.brand_model.loc[brand_model['brand'] == brand, 'models'].str.lower().unique().tolist()

            for model in sub_set_models:

                if (len(str(model)) < 2) | (not _is_strict_alnum(model)) | (len(str(model).split(" ")) < 2):

                    for w in x_list:

                        if w == model:

                            models.append({'brand':brand,'model':model})

                else:
                    if self._check(string,str(model)):

                        string = string.replace(str(model),'')

                        models.append({'brand':brand,'model':model})

            if any(brand in model for model in models):

                continue

            else:

                models.append({'brand':brand,'model':None})

        return models, string

    def _search_models(self,string,rem_num_only = True):
        """To search models from excel from the input string

        Returns: dataframe
            1. Models extracted Dataframe
        """

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

                if self._check(string,str(model)):

                    brands = self.brand_model.loc[brand_model['models'] == model, 'brand']

                    for brand in brands:

                        models.append({'brand':brand,'model':model})

        return models

    def extract_models(self,string,brands):
        """To Extract models from string using brand

        Returns: dataframe
            1. Models extracted Dataframe
        """

        all_models = []

        extracted_models, altered_string = (self._search_models_based_on_brand(string,brands))

        all_models.extend(extracted_models)

        #all_models.extend((self._search_models(altered_string)))

        extracted_dataframe = pd.DataFrame(all_models)

        if 'brand' not in extracted_dataframe.columns:
            extracted_dataframe['brand'] = np.nan
        if 'model' not in extracted_dataframe.columns:
            extracted_dataframe['model'] = np.nan

        extracted_dataframe['user_id']= self.user_id

        extracted_dataframe['message_id']= self.message_id

        extracted_dataframe['tag']= self.tag

        if 'date' in self.input_object:

            trans_date = self.input_object['date']

        else:

            trans_date = date.today()

        extracted_dataframe['date']= str(trans_date)

        extracted_dataframe = extracted_dataframe[['user_id','date','tag','brand','model','message_id']]

        extracted_dataframe = extracted_dataframe.dropna()

        extracted_dataframe = extracted_dataframe.drop_duplicates()

        return extracted_dataframe

    def extract_brand_model(self):
        """To Extract brand and model from the text

        Returns: dataframe
            1. Models and brand extracted Dataframe
        """

        preprocessed_text = self.pre_processing()

        tokens = self._stop_words_removal(preprocessed_text)

        all_brands = self.brand_model['brand'].str.lower().unique().tolist()

        brands_list = self._search_brands(all_brands,tokens)

        extracted_dataframe = self.extract_models(preprocessed_text,brands_list)

        return extracted_dataframe

    def get_pincode(self):
        """To get the pincode of the shop of the user

        Returns: Int
            1. Users Pincode
        """

        pin_code_query = "SELECT pin FROM {} WHERE user_id = {};".format(self.shop_table,self.user_id)

        pin_code_result = self.db_engine.execute(pin_code_query)

        pin_code_list = [r[0] for r in pin_code_result]

        return pin_code_list[0]

    def _generate_pincodes(self,pincode,num=5):
        """To generate pincodes based on pincode and range of number

        Returns: List
            1. Pincodes
        """
        pincodes = []

        for i in range(pincode-num,pincode):

            pincodes.append(i)


        for j in range(pincode,pincode+num):

            pincodes.append(j)

        return tuple(pincodes)

    def get_matching_user(self,pincodes):

        """To get users Id from database based on pincodes

        Returns: List
            1. User Ids
        """

        user_id_query = "SELECT user_id FROM {} WHERE pin IN {};".format(self.shop_table,pincodes)

        user_id_result = self.db_engine.execute(user_id_query)

        user_id_list = [r[0] for r in user_id_result]

        return user_id_list



    def _dataframe_available(self,user_id_list):

        """Get the existing model and brand based on userId from the database

        Returns: Dataframe
            1. Models and brands with userId
        """

        if self.tag =='sell':

            tag = 'buy'

        elif self.tag =='buy':

            tag = 'sell'

        if len(user_id_list)>1:

            extracted_model_query = "SELECT * from {} WHERE user_id IN {} AND tag = '{}' AND {}>DATE_SUB(now(), {});".format(self.ce_table,
                                                        tuple(user_id_list),tag,self.DATE_COLUMN,self.INTERVAL)

        elif len(user_id_list)==1:

            extracted_model_query = "SELECT * from {} WHERE user_id = {} AND tag = '{}' AND {}>DATE_SUB(now(), {});".format(self.ce_table,
                                                                        user_id_list[0],tag,self.DATE_COLUMN,self.INTERVAL)

        available_models = pd.read_sql_query(extracted_model_query,self.db_engine)

        return available_models



    # def get_existing_df(self):

    #     """Get the existing table from the database to be updated

    #     Returns: Dataframe
    #         1. Models and brands
    #     """

    #     existing_df_query = "SELECT * FROM {};".format(self.ce_table) #----table names in catalog

    #     existing_df = pd.read_sql_query(existing_df_query,self.db_engine)

    #     return existing_df



    def uploaded_dateframe(self,dataframe):

        """Upload dataframe to database and return status of upload
        Returns: Boolean
        """

        try:

            dataframe.to_sql(self.ce_table, con=self.db_engine, if_exists='append',index=False)

            return True

        except Exception as e:

            print(str(e))

            return False



    def get_users(self,extracted_dataframe,available_models):

        """Get final dataframe of users matching the brand and model extracted from the
        text and returns users list

        Returns: List
            1. Final Users
        """

        final_df = pd.merge(extracted_dataframe, available_models, on=['brand','model'], how='inner')

        if final_df.empty:

            final_df = pd.merge(extracted_dataframe, available_models, on=['brand'], how='inner')

            if final_df.empty:

                final_df = available_models

        final_list = final_df['user_id'].tolist()

        return final_list



    # def get_upload_final_df(self,existing_df,extracted_dataframe):
    #     """Get dataframe to be updated in the database

    #     Returns: Dataframe
    #         1. Models and brands with userId
    #     """

    #     if existing_df.empty:
    #         upload_df_final = extracted_dataframe

    #     else:
    #         upload_df = pd.concat([existing_df,extracted_dataframe])

    #         upload_df_final = upload_df.drop_duplicates()

    #     return upload_df_final




    def available_user_with_model(self,pincode):
        """Generate pincodes based on user's pincode and get matching users

        Returns: Dataframe
            1. Models and brands with userId
        """
        available_dataframe = pd.DataFrame()

        num = 5

        while(available_dataframe.empty)and(num<self.iterate_limit):

            print(num)

            pincodes_tuple = self._generate_pincodes(pincode,num)

            matching_users = self.get_matching_user(pincodes_tuple)

            matching_users = list(set(matching_users))

            available_dataframe = self._dataframe_available(matching_users)

            if num == 5:

                num = 100

            elif num == 100:

                num = 1000

            else:

                num += 1000

        return available_dataframe



    def get_users_available(self,available_dataframe,extracted_dataframe):
        """Get matching users based on model and brand extracted from text and matching
        data from the database

        Returns: JSON
            1. UserIds
        """

        extracted_brand_model_df = extracted_dataframe[['brand','model']]

        matched_users = self.get_users(extracted_brand_model_df,available_dataframe)

        user_dict = {}

        for i in range(len(set(matched_users))):
            if matched_users[i]!=self.user_id:
                user_dict[i] = matched_users[i]

        return json.dumps(user_dict)

    def matched_users(self):
        """
        Runs extraction process from start to end

        This involves:
            1. Preprocess text
            2. Extract brand, models, issues and tools from the processed text
            3. Updates database tables with extracted info
            4. Generate Pincode based on users pincode
            5. Extract users based on pincode,model,brand and tag
        returns:
            JSON object of matching users ID.
        """
        try:
            extracted_dataframe = self.extract_brand_model()

            #existing_df = self.get_existing_df()

            #upload_df_final = self.get_upload_final_df(existing_df,extracted_dataframe)

            upload_status = self.uploaded_dateframe(extracted_dataframe)

            pincode = self.get_pincode()

            if upload_status:

                available_dataframe = self.available_user_with_model(pincode)

                user_json = self.get_users_available(available_dataframe,extracted_dataframe)

                return user_json
        except:
            return {}
