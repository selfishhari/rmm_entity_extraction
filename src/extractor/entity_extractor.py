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
from src.database_utils.database_config import DataBaseEngine
from src.extractor.fuzz_match import FuzzySearch
from src.extractor.service_extractor import ServiceExtractor
from geopy import Point
from geopy.distance import geodesic




def _is_strict_alnum(x):

    if str(x).isdigit():
        return False

    o_len = len(str(x))

    remove_digits = str.maketrans('', '', string.digits)

    procx = str(x).translate(remove_digits)

    p_len = len(procx)

    return o_len > p_len

class MatchUser():

    def __init__(self,catalog,credentials):
        try:

            self.catalog = vars(catalog)

            self.credentials = vars(credentials)

            db_config = DataBaseEngine(self.catalog,self.credentials)

            self.db_engine = db_config.config_engine()

            self.alternative_name = pd.read_csv(self.catalog['alternative_name_csv'])

            self.brand_model = pd.read_csv(self.catalog['brand_model_csv'])

            self.brand_rank = pd.read_csv(self.catalog['brand_rank_csv'])

            self.shop_table = self.catalog['SHOP_TABLE']

            self.ce_table = self.catalog['CUSTOMER_ENTITY_TABLE']

            self.INTERVAL = self.catalog['INTERVAL']

            self.iterate_limit = self.catalog['ITERATE_LIMIT']

            self.iterate_limit_kms = self.catalog['ITERATE_LIMIT_KMS']

            self.DATE_COLUMN = self.catalog['DATE_COLUMN']

            self.service_extractor = ServiceExtractor(self.catalog)

            print("Done")
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

                # Doing exact match if length of model is less than 2 or model is strictly a digit(numeric) and string is a single word
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

                break

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

            #if there is a match, dataframe of matched string is formed
            if len(matching_word) > 0:

                df_matching = pd.DataFrame(matching_word)

                extracted_dataframe = pd.concat([extracted_dataframe,df_matching])

        #predicting service using naive bayes model
        predicted_service = self.service_extractor.predict_label(string)

        print("predicted service",predicted_service)

        extracted_dataframe['service'] = predicted_service

        print("@@@@")
        #creating dataframe
        extracted_dataframe['user_id']= self.user_id

        extracted_dataframe['message_id']= self.message_id

        extracted_dataframe['tag']= self.tag

        if 'date' in self.input_object:

            trans_date = self.input_object['date']

        else:

            trans_date = date.today()

        extracted_dataframe['date']= str(trans_date)

        extracted_dataframe = extracted_dataframe[['user_id','date','tag','brand','model','message_id','service']]

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


    def get_lat_long_user(self):
        """To get the latitude and longitude of the shop of the user

        Returns: String
            1. latitude, longitude
        """

        lat_long_query = "SELECT {},{} FROM {} WHERE {} = {};".format(self.catalog['LATITUDE_COLUMN'],
                                self.catalog['LONGITUDE_COLUMN'],self.shop_table,self.catalog['USER_ID_COLUMN'],self.user_id)

        lat_long_result = pd.read_sql_query(lat_long_query,self.db_engine)

        print(lat_long_result)
        latitude = lat_long_result.loc[0,'latitude']

        longitude = lat_long_result.loc[0,'longitude']

        return latitude,longitude

    def _get_min_max_lat_long(self,east,north,south,west):
        """Get minimum and maximum latitude and longitude based on directions

        Returns: Dict
            1. Lat Long Dict"""
        lat_long = {}

        latitude_list = []

        longitude_list = []

        #splitting string into list and appending latitude and longitude to separate list
        east_lat = float(east.split(", ")[0])

        east_long = float(east.split(", ")[1])

        latitude_list.append(east_lat)

        longitude_list.append(east_long)


        north_lat = float(north.split(", ")[0])

        north_long = float(north.split(", ")[1])

        latitude_list.append(north_lat)

        longitude_list.append(north_long)


        south_lat = float(south.split(", ")[0])

        south_long = float(south.split(", ")[1])

        latitude_list.append(south_lat)

        longitude_list.append(south_long)


        west_lat = float(west.split(", ")[0])

        west_long = float(west.split(", ")[1])

        latitude_list.append(west_lat)

        longitude_list.append(west_long)

        #finding minimum and maximum of latitude and longitude and assigning it to dict
        lat_long['lat_min'] = min(latitude_list)

        lat_long['lat_max'] = max(latitude_list)

        lat_long['long_min'] = min(longitude_list)

        lat_long['long_max'] = max(longitude_list)

        print(lat_long)
        return lat_long

    def _generate_lat_long(self,latitude,longitude,kms=5):
        """To generate latitude and longitude based on lat long of user and range of kilometer

        Returns: List
            1. Lat longs
        """
        east = geodesic(kilometers=kms).destination(Point(latitude, longitude), 90).format_decimal()

        north = geodesic(kilometers=kms).destination(Point(latitude, longitude), 0).format_decimal()

        south = geodesic(kilometers=kms).destination(Point(latitude, longitude), 180).format_decimal()

        west = geodesic(kilometers=kms).destination(Point(latitude, longitude), 270).format_decimal()

        lat_long = self._get_min_max_lat_long(east, north, south, west)

        return lat_long

    def get_matching_user_by_lat_long(self,lat_long):

        """To get users Id from database based on lat long

        Returns: List
            1. User Idslat_max
        """
        lat_min = lat_long['lat_min']

        lat_max = lat_long['lat_max']

        long_min = lat_long['long_min']

        long_max = lat_long['long_max']

        user_id_query = "SELECT {} FROM {} WHERE {} BETWEEN {} AND {} AND {} BETWEEN {} and {}".format(self.catalog['USER_ID_COLUMN'],
                                            self.shop_table,self.catalog['LATITUDE_COLUMN'],
                                            lat_min,lat_max,self.catalog['LONGITUDE_COLUMN'],long_min,long_max)

        user_id_result = self.db_engine.execute(user_id_query)

        user_id_list = [r[0] for r in user_id_result]

        #removing the user id of the customer
        user_id_list.remove(self.user_id)

        print(user_id_list)

        return user_id_list



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

        print(available_models)
        return available_models

    def _dataframe_available_all_value(self):

        """Get the existing model and brand based on userId from the database

        Returns: Dataframe
            1. Models and brands with userId
        """

        #initializing extract tag
        if self.tag =='sell':

            tag = 'buy'

        elif self.tag =='buy':

            tag = 'sell'

        extracted_model_query = "SELECT * from {} WHERE tag = '{}' AND {}>DATE_SUB(now(), {});".format(self.ce_table,
                                            tag,self.DATE_COLUMN,self.INTERVAL)

        available_models = pd.read_sql_query(extracted_model_query,self.db_engine)

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

    def get_matching_dataframe(self,extracted_dataframe,available_models):

        """Get final dataframe of users matching the brand and model extracted from the
        text and returns messages list

        Returns: List
            1. Message Ids
        """
        #chnaging dtype of columns
        extracted_dataframe = extracted_dataframe[['brand','model','service']]

        extracted_dataframe['brand'] = extracted_dataframe[['brand']].astype(str)

        extracted_dataframe['model'] = extracted_dataframe[['model']].astype(str)

        extracted_dataframe['service'] = extracted_dataframe[['service']].astype(str)

        available_models['brand'] = available_models[['brand']].astype(str)

        available_models['model'] = available_models[['model']].astype(str)

        available_models['service'] = available_models[['service']].astype(str)

        # Merging on both models and brands
        final_df = pd.merge(extracted_dataframe, available_models, on=['brand','service'], how='inner')

        if final_df.empty:

            # If the dataframe is empty, then merging on brand

            final_df = pd.merge(extracted_dataframe, available_models, on=['brand'], how='inner')

            # Still, if its empty merging, returning all
            if final_df.empty:

                final_df = available_models

        print(final_df.columns)
        return final_df

    def available_user_with_model_matching(self,latitude,longitude,matching_dataframe):
        """Generate pincodes based on user's latitude and lonigtude and get matching users

        Returns: Dataframe
            1. Models and brands with userId
        """
        available_dataframe = pd.DataFrame()

        kms = 5

        #Iterating till there is no empty datadframe
        while(available_dataframe.empty)and(kms<self.iterate_limit_kms):

            #getting pincodes

            lat_long = self._generate_lat_long(latitude,longitude,kms)

            #Identifying matching users available in the pincodes
            matching_users = self.get_matching_user_by_lat_long(lat_long)

            matching_users = list(set(matching_users))

            available_dataframe = matching_dataframe[matching_dataframe['user_id'].isin(matching_users)]

            #increasing number of pincodes to form

            kms += 5

        return available_dataframe

    def get_users_available(self,available_dataframe):
        """Get matching users based on model and brand extracted from text and matching
        data from the database

        Returns: JSON
            1. UserIds
        """

        #extracting final set of user matching our requirements
        matched_users = available_dataframe['user_id'].tolist()

        matched_users = list(set(matched_users))

        user_dict = {}

        for i in range(len(matched_users)):
            user_dict[i] = matched_users[i]

        return json.dumps(user_dict)

    def get_messages_available(self,available_dataframe):
        """Get matching message ids based on model and brand extracted from text and matching
        data from the database

        Returns: JSON
            1. MessageIds
        """

        #extracting final set of user matching our requirements

        message_ids = available_dataframe['message_id'].tolist()

        message_ids = list(set(message_ids))

        message_id_dict = {}

        for i in range(len(message_ids)):
            message_id_dict[i] = message_ids[i]

        return json.dumps(message_id_dict)


    def initialize_text(self,input_object):
        self.input_object = input_object

        self.tag = self.input_object['tag']

        self.user_id = self.input_object['id']

        self.text = self.input_object['text']

        self.message_id = self.input_object['message_id']

        return True

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
           # preprocessing the text
            pre_processed_text = self.pre_processing(self.text)

            #extracting model brand and service from the text
            extracted_dataframe = self.extract_brand_model(pre_processed_text)

            #uploading the extracted dataframe
            upload_status = self.uploaded_dateframe(extracted_dataframe)

            #getting latitude and longitude of the user
            latitude,longitude = self.get_lat_long_user()

            #getting the dataframe of available model brand and service
            available_dataframe = self._dataframe_available_all_value()
            print(available_dataframe)

            #matching the available brand model and service with extracted dataframe
            matching_brand_service = self.get_matching_dataframe(extracted_dataframe,available_dataframe)
            print(matching_brand_service)

            #upload status defines whether the extracted dataframe is successfully uploaded in the database
            if upload_status:

                #getting the available user from matched messages according to users location
                available_dataframe_final = self.available_user_with_model_matching(latitude,longitude,matching_brand_service)
                print(available_dataframe_final)

                user_json = self.get_users_available(available_dataframe_final)

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
            # preprocessing the text
            pre_processed_text = self.pre_processing(self.text)

            #extracting model brand and service from the text
            extracted_dataframe = self.extract_brand_model(pre_processed_text)

            #uploading the extracted dataframe
            upload_status = self.uploaded_dateframe(extracted_dataframe)

            #getting latitude and longitude of the user
            latitude,longitude = self.get_lat_long_user()

            #getting the dataframe of available model brand and service
            available_dataframe = self._dataframe_available_all_value()
            print(available_dataframe)

            #matching the available brand model and service with extracted dataframe
            matching_brand_service = self.get_matching_dataframe(extracted_dataframe,available_dataframe)
            print(matching_brand_service)

            #upload status defines whether the extracted dataframe is successfully uploaded in the database
            if upload_status:

                #getting the available user from matched messages according to users location
                available_dataframe_final = self.available_user_with_model_matching(latitude,longitude,matching_brand_service)
                print(available_dataframe_final)

                #getting the message ids
                message_json = self.get_messages_available(available_dataframe_final)

                return message_json


        except Exception as e:
            print(str(e))
            return {}
