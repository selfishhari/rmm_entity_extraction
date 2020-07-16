import pandas as pd
from created import MatchUser


class AccuracyMeasure():
    def __init__(self,test_dataframe):
        self.test_set = test_dataframe

    def preprocessing(self):
        self.test_set["model "]= self.test_set["model "].str.split("|")
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

    def accuracy_score_brands(self,y_true,y_pred):
        accuracy_score = 0
        for i in range(len(y_true)):
            if y_true[i]==y_pred[i]:
                accuracy_score =+ 1
        total_accuracy = (accuracy_score/len(y_true))
        return total_accuracy

    def temp_brand_accuracy_score(self,y_true,y_pred):
        print("TEMPPPPPPPPPPPPPPPPPP")
        accuracy_score = 0
        for i in range(len(y_true)):
            # print(y_true[i])
            # print(y_pred[i])
            if y_true[i] in y_pred[i]:
                accuracy_score += 1
            print(accuracy_score)

        total_accuracy = (accuracy_score/len(y_true))
        return total_accuracy

    def calculate_accuracy(self):
        self.preprocessing()
        self.test_set["brand_pred"] = "1"
        self.test_set["model_pred"] = "1"
        for index, row in self.test_set.iterrows():
            question = row['question ']
            print(question)
            input_object = {'id':1,'text':question,'tag':'buy','user_id':543,'message_id':6}
            match_user = MatchUser(input_object)
            df = match_user.extract_brand_model()
            print(df)
            brand = df['brand'].tolist()
            models = df['model'].tolist()
            print(models)
            self.test_set.at[index,'brand_pred'] = brand
            self.test_set.at[index,'model_pred'] = models

        self.test_set.dropna(inplace=True)
        self.test_set.reset_index(drop=True, inplace=True)

        print(self.test_set.tail())
        model_accuracy = self.accuracy_score_models(self.test_set['model '],self.test_set['model_pred'])
        brand_accuracy = self.temp_brand_accuracy_score(self.test_set['brand '],self.test_set['brand_pred'])

        return model_accuracy,brand_accuracy
