import pandas as pd
import sys
sys.path.append("/home/ec2-user/rmm_entity_extraction")
from src.extractor.entity_extractor import MatchUser
from conf.public import catalog, credentials


test_brand_output = catalog.TEST_OUTPUT_BRAND_CSV

test_model_output = catalog.TEST_OUTPUT_MODEL_CSV

class AccuracyMeasure():
    def __init__(self,test_set_brand,test_set_model):

        self.catalog = catalog

        self.credentials = credentials

        self.test_set_brand = test_set_brand

        self.test_set_model = test_set_model

        self.brand_output = test_brand_output

        self.model_output = test_model_output

    def preprocessing(self):
        #preprocessing model and brands dataset

        self.test_set_model["model "]= self.test_set_model["model "].str.split("|")

        self.test_set_brand["brand"]= self.test_set_brand["brand"].str.split(",")

        return True

    def accuracy_score_models(self,y_true,y_pred):
        """y_true and y_pred is a list consisting of model names
        y_true = ['note','pro', 'j7']
        y_pred = ['note','pro'] """
        accuracy_score = 0

        #iterating over lists
        for i in range(len(y_true)):

            #finding maximum length

            if len(y_true[i])>len(y_pred[i]):

                length = len(y_true[i])

            else:

                length = len(y_pred[i])

            #finding common elements
            common_elements = list(set(y_true[i]) & set(y_pred[i]))

            #finding accuracy of one element
            single_accuracy = (len(common_elements)/length)

            #adding it to total accuracy
            accuracy_score += single_accuracy

            print(accuracy_score)

        total_accuracy = (accuracy_score/len(y_true))

        return total_accuracy

    def accuracy_score_brand(self,y_true,y_pred):

        accuracy = 0

        for i in range(len(y_true)):
            #checking if element exist in true values
            if y_true[i] in y_pred:

                accuracy +=1

        return (accuracy/len(y_true))

    def accuracy_score_individual(self,y_true,y_pred):
        # Finding the biggest length

        if len(set(y_true))>len(set(y_pred)):

            length = len(set(y_true))

        else:

            length = len(set(y_pred))

        #finding common elements
        common_elements = list(set(y_true) & set(y_pred))

        #Finding single accuracy
        single_accuracy = (len(common_elements)/length)

        return single_accuracy

    def calculate_total_accuracy(self,accuracy_list):

        #totalling all accuracy
        total = accuracy_list.sum()

        length = len(accuracy_list)

        #getting final accuracy by dividing total accuracy with its length
        accuracy = total/length
        return accuracy

    def calculate_accuracy_model(self):

        #initializing column value
        self.test_set_model["model_pred"] = "1"

        for index, row in self.test_set_model.iterrows():

            question = row['question ']

            #forming input object
            input_object = {'id':1,'text':question,'tag':'buy','user_id':543,'message_id':6}

            #predicting brands and model
            match_user = MatchUser(self.catalog,self.credentials,input_object)

            preprocessed_text = match_user.pre_processing(match_user.text)

            df = match_user.extract_brand_model(preprocessed_text)

            #assigning models as list value to model_pred
            models = df['model'].tolist()

            models = [x for x in models if x is not None]

            models = list(set(models))

            self.test_set_model.at[index,'model_pred'] = models

        #Dropping null and resetting index
        self.test_set_model.dropna(inplace=True)

        self.test_set_model.reset_index(drop=True, inplace=True)

        #initializing accuracy column value
        self.test_set_model["accuracy"] = " "

        #iterating over rows and finding accuracy
        for index,row in self.test_set_model.iterrows():

            y_true = row['model ']

            y_pred = row['model_pred']

            accuracy = self.accuracy_score_individual(y_true,y_pred)

            self.test_set_model.at[index,'accuracy'] = accuracy

        model_accuracy = self.calculate_total_accuracy(self.test_set_model['accuracy'])

        return model_accuracy

    def calculate_accuracy_brand(self):

        #initializing column value
        self.test_set_brand["brand_pred"] = "1"

        for index, row in self.test_set_brand.iterrows():
            question = row['question ']

            #forming input object
            input_object = {'id':1,'text':question,'tag':'buy','user_id':543,'message_id':6}

            #predicting brands and model
            match_user = MatchUser(self.catalog,self.credentials,input_object)

            preprocessed_text = match_user.pre_processing(match_user.text)

            df = match_user.extract_brand_model(preprocessed_text)

            #assigning brand as list value to brand_pred
            brands = df['brand'].tolist()

            brands = list(set(brands))

            self.test_set_brand.at[index,'brand_pred'] = brands

        #Dropping null and resetting index value
        self.test_set_brand.dropna(inplace=True)

        self.test_set_brand.reset_index(drop=True, inplace=True)

        #initializing accuracy column value
        self.test_set_brand["accuracy"] = " "

        #iterating over rows and finding accuracy
        for index,row in self.test_set_brand.iterrows():

            y_true = row['brand']

            y_pred = row['brand_pred']

            accuracy = self.accuracy_score_brand(y_true,y_pred)

            self.test_set_brand.at[index,'accuracy'] = accuracy

        brand_accuracy = self.calculate_total_accuracy(self.test_set_brand['accuracy'])

        return brand_accuracy

    def save_model(self,dataframe,output_path):

        dataframe.to_csv(output_path,index=False)

        return True

    def calculate_accuracy(self):

        #preprocessing
        self.preprocessing()

        #calculating brand accuracy
        brand_accuracy = self.calculate_accuracy_brand()

        #calculating model accuracy
        model_accuracy = self.calculate_accuracy_model()

        #saving the models
        self.save_model(self.test_set_model,self.model_output)

        self.save_model(self.test_set_brand,self.brand_output)

        return model_accuracy, brand_accuracy