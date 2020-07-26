from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import json
import pandas as pd
import numpy as np
from datetime import date
import string
import operator
import sqlalchemy
import sys
sys.path.append("/home/ec2-user/rmm_entity_extraction")
from src.extractor.fuzz_match import FuzzySearch





def _is_strict_alnum(x):

    if str(x).isdigit():
        return False

    o_len = len(str(x))

    remove_digits = str.maketrans('', '', string.digits)

    procx = str(x).translate(remove_digits)

    p_len = len(procx)

    return o_len > p_len

class MatchUser():

    def __init__(self,catalog,credentials,input_object):
        try:
            self.input_object = input_object

            self.catalog = vars(catalog)

            self.credentials = vars(credentials)

            self.tag = self.input_object['tag']

            self.user_id = self.input_object['id']

            self.text = self.input_object['text']

            self.message_id = self.input_object['message_id']

            self.user_name = self.credentials['DB_USERNAME']

            self.password = self.credentials['DB_PASSWORD']

            self.host = self.catalog['DB_HOST']

            self.db_name =self.catalog['DB_NAME']

            self.db_engine = sqlalchemy.create_engine(
                'mysql+pymysql://{0}:{1}@{2}/{3}?charset=utf8mb4'.format(self.user_name,
                                                self.password,
                                                self.host,
                                                self.db_name))

            self.alternative_name = pd.read_csv(self.catalog['alternative_name_csv'])

            self.brand_model = pd.read_csv(self.catalog['brand_model_csv'])

            self.brand_rank = pd.read_csv(self.catalog['brand_rank_csv'])

            self.shop_table = self.catalog['SHOP_TABLE']

            self.ce_table = self.catalog['CUSTOMER_ENTITY_TABLE']

            self.INTERVAL = self.catalog['INTERVAL']

            self.iterate_limit = self.catalog['ITERATE_LIMIT']

            self.DATE_COLUMN = self.catalog['DATE_COLUMN']

        except Exception as e:
            print("Constructor class")
            print(str(e))


    def pre_processing(self,text):
        """
        Removes links, blanks, None, lowercasing, replacinh newline chars.

        Returns: string
            1. Preprocessed string
        """
        string_1 = text

        string_1 = re.sub("(?P<url>https?://[^\s]+)","", string_1)

        string_1 = string_1.lower()

        string_1 = string_1.replace("\n", " ")

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

        #exact search
        items = []

        for item in items_list:

            if (exact_search == True) | (len(item)<4):

                for w in x_list:

                    if item == w.strip():

                        items += [item]
                        #break;

            else:

                if item in x_list:

                    items += [item]

        #searching for alternative brand names
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

        #converting alternative brand name column to list
        alternative_name_copy['alt'] = alternative_name_copy['alt'].apply(lambda x: str(x).split(", "))

        alt_name = alternative_name_copy.set_index('brand')['alt'].to_dict()

        brand_name = []

        #checking if alternative name exists
        for item in x_list:

            for key, value in alt_name.items():

                if item.strip() in value:

                    brand_name.append(key)

        return brand_name

    def _check(self,string, sub_str):
        """To check if a sub string is present inside a string

        Returns: Boolean
        """

        if (str(string).find(str(sub_str)) == -1):

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

        #splitting the input string
        x_list = string.split(" ")

        for brand in brands:

            #getting mmodels based on brand
            sub_set_models = self.brand_model.loc[self.brand_model['brand'] == brand, 'models'].str.lower().unique().tolist()

            for model in sub_set_models:

                # Doing exact match if length of model is less than 2 or model is strictly a digit(numeric)
                if (len(str(model)) < 2) | (not _is_strict_alnum(model)) | (len(str(model).split(" ")) < 2):

                    #exact match
                    for w in x_list:

                        if w == model:

                            #removing brand name from string
                            string = string.replace(str(model),'')

                            models.append({'brand':brand,'model':model})

                else:
                    #substring match
                    if self._check(string,str(model)):

                        #removing brand name from string
                        string = string.replace(str(model),'')

                        models.append({'brand':brand,'model':model})

            #checking if model is available for the brand, if not assigning none(model) for that brand
            if any(brand in model for model in models):

                continue

            else:

                models.append({'brand':brand,'model':None})


        return models, string

    def _get_brand_rank(self,brand_list):

        sort_value = {}

        for brand in brand_list:

            rank = self.brand_rank.loc[self.brand_rank['brand'] == brand, 'brand_rank']

            if rank.any():

                rank = rank.iloc[0]

            else:

                rank = 1000

            sort_value[brand]=rank

        print(sort_value)

        most_popular = min(sort_value,key=sort_value.get)

        return most_popular

    def _search_models(self,string,rem_num_only = True):
        """To search models from excel from the input string

        Returns: dataframe
            1. Models extracted Dataframe
        """

        #extracting all the models
        all_models = self.brand_model['models'].str.lower().unique().tolist()

        models= []

        if rem_num_only:

            for item in all_models:

                #removing digits
                if str(item).isdigit():

                    all_models.remove(item)

        #splitting the altered string
        x_list = string.split(" ")

        for model in all_models:

            if (len(str(model)) < 2) | (not _is_strict_alnum(model)) | (len(str(model).split(" ")) < 2):

                #exact match
                for w in x_list:

                    if w == model:

                        brands = self.brand_model.loc[self.brand_model['models'] == model, 'brand']

                        popular_brand = self._get_brand_rank(brands)

                        models.append({'brand':popular_brand,'model':model})

            else:

                if self._check(string,str(model)):

                    #substring match
                    brands = self.brand_model.loc[self.brand_model['models'] == model, 'brand']

                    popular_brand = self._get_brand_rank(brands)

                    models.append({'brand':popular_brand,'model':model})

        return models

    def extract_models(self,string,brands):
        """To Extract models from string using brand

        Returns: dataframe
            1. Models extracted Dataframe
        """

        all_models = []

        #extracting models based on brands
        extracted_models, altered_string = (self._search_models_based_on_brand(string,brands))

        all_models.extend(extracted_models)

        #extracting models based on substrings
        all_models.extend((self._search_models(altered_string)))

        extracted_dataframe = pd.DataFrame(all_models)

        if 'brand' not in extracted_dataframe.columns:
            extracted_dataframe['brand'] = np.nan
        if 'model' not in extracted_dataframe.columns:
            extracted_dataframe['model'] = np.nan

        if extracted_dataframe.empty:
            #Doing direct fuzz match
            words = self.brand_model['brand'].str.lower().unique().tolist()

            spell_check = FuzzySearch(words)

            spell_check.check(string)

            matching_word = spell_check.suggestions()
            print("2")
            #if there is a match, dataframe of matched string is formed
            if len(matching_word) > 0:

                df_matching = pd.DataFrame(matching_word)
                print("3")
                print(matching_word)
                extracted_dataframe = pd.concat([extracted_dataframe,df_matching])
                print(extracted_dataframe)
        #creating dataframe
        extracted_dataframe['user_id']= self.user_id

        extracted_dataframe['message_id']= self.message_id

        extracted_dataframe['tag']= self.tag

        if 'date' in self.input_object:

            trans_date = self.input_object['date']

        else:

            trans_date = date.today()

        extracted_dataframe['date']= str(trans_date)

        extracted_dataframe = extracted_dataframe[['user_id','date','tag','brand','model','message_id']]

        #Dropping duplicates
        extracted_dataframe = extracted_dataframe.drop_duplicates()

        print(extracted_dataframe)

        return extracted_dataframe

    def remove_brand_name(self,dataframe):

        for index, row in dataframe.iterrows():

            brand = row['brand']

            model = row['model']

            #checking for none value
            if brand is not None and model is not None:
                check_status = self._check(model,brand)

                if check_status:

                    # replacing brand name in models with space
                    model = model.replace(str(brand),'')

                    #removing unwantes spaces
                    model = model.lstrip()

                    dataframe.at[index,'model'] = model

        dataframe = dataframe.drop_duplicates()

        return dataframe

    def extract_brand_model(self,preprocessed_text):
        """To Extract brand and model from the text

        Returns: dataframe
            1. Models and brand extracted Dataframe
        """

        #Forming tokens
        tokens = self._stop_words_removal(preprocessed_text)

        #Getting all brand names from dataframe
        all_brands = self.brand_model['brand'].str.lower().unique().tolist()

        #Extracting matching brand name
        brands_list = self._search_brands(all_brands,tokens)

        #Extracting models
        extracted_dataframe = self.extract_models(preprocessed_text,brands_list)

        #Removing brand names from models
        extracted_dataframe = self.remove_brand_name(extracted_dataframe)

        return extracted_dataframe

    def get_pincode(self):
        """To get the pincode of the shop of the user

        Returns: Int
            1. Users Pincode
        """

        pin_code_query = "SELECT {} FROM {} WHERE {} = {};".format(self.catalog['PIN_COLUMN'],
                                    self.shop_table,self.catalog['USER_ID_COLUMN'],self.user_id)

        pin_code_result = self.db_engine.execute(pin_code_query)

        pin_code_list = [r[0] for r in pin_code_result]

        return pin_code_list[0]

    def _generate_pincodes(self,pincode,num=5):
        """To generate pincodes based on pincode and range of number

        Returns: List
            1. Pincodes
        """
        pincodes = []

        #negative range of pincode
        for i in range(pincode-num,pincode):

            pincodes.append(i)

        #positive range of pincode
        for j in range(pincode,pincode+num):

            pincodes.append(j)

        return tuple(pincodes)

    def get_matching_user(self,pincodes):

        """To get users Id from database based on pincodes

        Returns: List
            1. User Ids
        """

        user_id_query = "SELECT {} FROM {} WHERE {} IN {};".format(self.catalog['USER_ID_COLUMN'],
                                     self.shop_table,self.catalog['PIN_COLUMN'],pincodes)

        user_id_result = self.db_engine.execute(user_id_query)

        user_id_list = [r[0] for r in user_id_result]

        #removing the user id of the customer
        user_id_list.remove(self.user_id)

        print(user_id_list)

        return user_id_list

    # def _generate_lat_long(self,pincode,km=15):
    #     """To generate latitude and longitude based on lat long of user and range of kilometer

    #     Returns: List
    #         1. Lat longs
    #     """
    #     lat_long = {}

    #     #negative range of pincode
    #     for i in range(pincode-num,pincode):

    #         pincodes.append(i)

    #     #positive range of pincode
    #     for j in range(pincode,pincode+num):

    #         pincodes.append(j)

    #     return lat_long

    # def get_matching_user_by_lat_long(self,lat_long):

    #     """To get users Id from database based on lat long

    #     Returns: List
    #         1. User Ids
    #     """
    #     lat1 = lat_long['lat_1']

    #     lat2 = lat_long['lat_2']

    #     long1 = lat_long['long_1']

    #     long2 = lat_long['long_2']

    #     user_id_query = "SELCT {} FROM {} WHERE {} BETWEEN {} AND {} AND {} BETWEEN {} and {}".format(self.catalog['USER_ID_COLUMN'],
    #                                         self.shop_table,self.catalog['LATITUDE_COLUMN'],
    #                                         lat1,lat2,self.catalog['LONGITUDE_COLUMN'],long1,long2)

    #     user_id_result = self.db_engine.execute(user_id_query)

    #     user_id_list = [r[0] for r in user_id_result]

    #     #removing the user id of the customer
    #     user_id_list.remove(self.user_id)

    #     print(user_id_list)

    #     return user_id_list



    def _dataframe_available(self,user_id_list):

        """Get the existing model and brand based on userId from the database

        Returns: Dataframe
            1. Models and brands with userId
        """

        #initializing extract tag
        if self.tag =='sell':

            tag = 'buy'

        elif self.tag =='buy':

            tag = 'sell'

        if len(user_id_list)>0:

            if len(user_id_list)>1:

                extracted_model_query = "SELECT * from {} WHERE {} IN {} AND tag = '{}' AND {}>DATE_SUB(now(), {});".format(self.ce_table,
                                                    self.catalog['USER_ID_COLUMN'],tuple(user_id_list),tag,self.DATE_COLUMN,self.INTERVAL)

            elif len(user_id_list)==1:

                extracted_model_query = "SELECT * from {} WHERE {} = {} AND tag = '{}' AND {}>DATE_SUB(now(), {});".format(self.ce_table,
                                                                self.catalog['USER_ID_COLUMN'],user_id_list[0],tag,self.DATE_COLUMN,self.INTERVAL)

            available_models = pd.read_sql_query(extracted_model_query,self.db_engine)
        else:
            available_models = pd.DataFrame()

        return available_models


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
        #chnaging dtype of columns
        extracted_dataframe['brand'] = extracted_dataframe[['brand']].astype(str)

        extracted_dataframe['model'] = extracted_dataframe[['model']].astype(str)

        available_models['brand'] = available_models[['brand']].astype(str)

        available_models['model'] = available_models[['model']].astype(str)

        # Merging on both models and brands
        final_df = pd.merge(extracted_dataframe, available_models, on=['brand','model'], how='inner')

        if final_df.empty:

            # If the dataframe is empty, then merging on brand

            final_df = pd.merge(extracted_dataframe, available_models, on=['brand'], how='inner')

            # Still, if its empty merging, returning all
            if final_df.empty:

                final_df = available_models

        final_list = final_df['user_id'].tolist()

        return final_list

    def get_messages(self,extracted_dataframe,available_models):

        """Get final dataframe of users matching the brand and model extracted from the
        text and returns messages list

        Returns: List
            1. Message Ids
        """
        #chnaging dtype of columns
        extracted_dataframe['brand'] = extracted_dataframe[['brand']].astype(str)

        extracted_dataframe['model'] = extracted_dataframe[['model']].astype(str)

        available_models['brand'] = available_models[['brand']].astype(str)

        available_models['model'] = available_models[['model']].astype(str)

        # Merging on both models and brands
        final_df = pd.merge(extracted_dataframe, available_models, on=['brand','model'], how='inner')

        if final_df.empty:

            # If the dataframe is empty, then merging on brand

            final_df = pd.merge(extracted_dataframe, available_models, on=['brand'], how='inner')

            # Still, if its empty merging, returning all
            if final_df.empty:

                final_df = available_models

        final_list = final_df['message_id'].tolist()

        return final_list


    def available_user_with_model(self,pincode):
        """Generate pincodes based on user's pincode and get matching users

        Returns: Dataframe
            1. Models and brands with userId
        """
        available_dataframe = pd.DataFrame()

        num = 5

        #Iterating till there is no empty datadframe
        while(available_dataframe.empty)and(num<self.iterate_limit):

            #getting pincodes

            pincodes_tuple = self._generate_pincodes(pincode,num)

            #Identifying matching users available in the pincodes
            matching_users = self.get_matching_user(pincodes_tuple)

            matching_users = list(set(matching_users))

            available_dataframe = self._dataframe_available(matching_users)

            #increasing number of pincodes to form
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

        #extracting final set of user matching our requirements
        matched_users = self.get_users(extracted_brand_model_df,available_dataframe)

        user_dict = {}

        for i in range(len(set(matched_users))):
            user_dict[i] = matched_users[i]

        return json.dumps(user_dict)

    def get_messages_available(self,available_dataframe,extracted_dataframe):
        """Get matching users based on model and brand extracted from text and matching
        data from the database

        Returns: JSON
            1. MessageIds
        """

        extracted_brand_model_df = extracted_dataframe[['brand','model']]

        #extracting final set of user matching our requirements
        matched_messages = self.get_messages(extracted_brand_model_df,available_dataframe)

        message_id_dict = {}

        for i in range(len(set(matched_messages))):
            message_id_dict[i] = matched_messages[i]

        return json.dumps(message_id_dict)




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
            pre_processed_text = self.pre_processing(self.text)

            extracted_dataframe = self.extract_brand_model(pre_processed_text)

            upload_status = self.uploaded_dateframe(extracted_dataframe)

            pincode = self.get_pincode()

            #upload status defines whether the extracted dataframe is successfully uploaded in the database
            if upload_status:

                available_dataframe = self.available_user_with_model(pincode)

                user_json = self.get_users_available(available_dataframe,extracted_dataframe)

                return user_json

        except Exception as e:
            print(str(e))

            return {}

    def get_matched_messages(self):
        """
        Runs extraction process from start to end

        This involves:
            1. Preprocess textuser_dict
            2. Extract brand, models, issues and tools from the processed text
            3. Updates database tables with extracted info
            4. Generate Pincode based on users pincode
            5. Extract users based on pincode,model,brand and tag
        returns:
            JSON object of matching users ID.
        """
        try:
            pre_processed_text = self.pre_processing(self.text)

            extracted_dataframe = self.extract_brand_model(pre_processed_text)

            upload_status = self.uploaded_dateframe(extracted_dataframe)

            pincode = self.get_pincode()

            #upload status defines whether the extracted dataframe is successfully uploaded in the database
            if upload_status:

                available_dataframe = self.available_user_with_model(pincode)

                message_json = self.get_messages_available(available_dataframe,extracted_dataframe)

                return message_json


        except Exception as e:
            print(str(e))
            return {}