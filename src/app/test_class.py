import pandas as pd
from created import MatchUser
from conf.public import catalog

test_brand_output = catalog.TEST_OUTPUT_BRAND_CSV

test_model_output = catalog.TEST_OUTPUT_MODEL_CSV

class AccuracyMeasure():
    def __init__(self,test_set_brand,test_set_model):
        self.test_set_brand = test_set_brand
        self.test_set_model = test_set_model
        self.brand_output = test_brand_output
        self.model_output = test_model_output

    def preprocessing(self):
        self.test_set_model["model "]= self.test_set_model["model "].str.split("|")
        self.test_set_brand["brand"]= self.test_set_brand["brand"].str.split(",")
        return True

    def accuracy_score_models(self,y_true,y_pred):
        accuracy_score = 0
        for i in range(len(y_true)):
            if len(y_true[i])>len(y_pred[i]):
                length = len(y_true[i])
            else:
                length = len(y_pred[i])

            common_elements = list(set(y_true[i]) & set(y_pred[i]))
            single_accuracy = (len(common_elements)/length)
            accuracy_score += single_accuracy
            print(accuracy_score)

        total_accuracy = (accuracy_score/len(y_true))
        return total_accuracy

    def accuracy_score_individual(self,y_true,y_pred):
        if len(set(y_true))>len(set(y_pred)):
            length = len(set(y_true))
        else:
            length = len(set(y_pred))

        common_elements = list(set(y_true) & set(y_pred))
        single_accuracy = (len(common_elements)/length)
        return single_accuracy

    def calculate_total_accuracy(self,accuracy_list):
        total = accuracy_list.sum()
        length = len(accuracy_list)
        accuracy = total/length
        return accuracy

    def calculate_accuracy_model(self):
        self.test_set_model["model_pred"] = "1"

        for index, row in self.test_set_model.iterrows():
            question = row['question ']
            print(question)
            input_object = {'id':1,'text':question,'tag':'buy','user_id':543,'message_id':6}
            match_user = MatchUser(input_object)
            df = match_user.extract_brand_model()
            print(df)
            models = df['model'].tolist()
            print(models)
            self.test_set_model.at[index,'model_pred'] = models

        self.test_set_model.dropna(inplace=True)
        self.test_set_model.reset_index(drop=True, inplace=True)

        self.test_set_model["accuracy"] = " "
        for index,row in self.test_set_model.iterrows():
            y_true = row['model ']
            y_pred = row['model_pred']
            accuracy = self.accuracy_score_individual(y_true,y_pred)
            self.test_set_model.at[index,'accuracy'] = accuracy


        #self.test_set.to_csv("/home/viky/Desktop/freelancer/FLASK/Predicted.csv")
        model_accuracy = self.calculate_total_accuracy(self.test_set_model['accuracy'])

        return model_accuracy

    def calculate_accuracy_brand(self):
        self.test_set_brand["brand_pred"] = "1"

        for index, row in self.test_set_brand.iterrows():
            question = row['question ']
            print(question)
            input_object = {'id':1,'text':question,'tag':'buy','user_id':543,'message_id':6}
            match_user = MatchUser(input_object)
            df = match_user.extract_brand_model()
            print(df)
            models = df['brand'].tolist()
            print(models)
            self.test_set_brand.at[index,'brand_pred'] = models

        self.test_set_brand.dropna(inplace=True)
        self.test_set_brand.reset_index(drop=True, inplace=True)

        self.test_set_brand["accuracy"] = " "
        for index,row in self.test_set_brand.iterrows():
            y_true = row['brand']
            y_pred = row['brand_pred']
            accuracy = self.accuracy_score_individual(y_true,y_pred)
            self.test_set_brand.at[index,'accuracy'] = accuracy

        brand_accuracy = self.calculate_total_accuracy(self.test_set_brand['accuracy'])

        return brand_accuracy

    def save_model(self,dataframe,output_path):

        dataframe.to_csv(output_path,index=False)

        return True

    def calculate_accuracy(self):
        self.preprocessing()
        model_accuracy = self.calculate_accuracy_model()
        brand_accuracy = self.calculate_accuracy_brand()
        self.save_model(self.test_set_model,self.model_output)
        self.save_model(self.test_set_brand,self.brand_output)

        return model_accuracy, brand_accuracy