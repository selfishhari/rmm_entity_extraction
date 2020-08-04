import pandas as pd
import sys
sys.path.append("/home/viky/Desktop/freelancer/flask")
from conf.public import catalog, credentials
from src.extractor.service_extractor import ServiceExtractor

test_service_output = catalog.TEST_OUTPUT_SERVICE_CSV

test_service_input = catalog.TEST_SET_SERVICE


class TestService():

    def __init__(self):

        self.test_service_output = test_service_output

        self.test_service_input = pd.read_csv(test_service_input)

        self.catalog = vars(catalog)

        self.service_extractor = ServiceExtractor(self.catalog)

    def accuracy_score_individual(self,y_true,y_pred):

        if y_true == y_pred:
            accuracy = 1
        else:
            accuracy = 0

        return accuracy

    def calculate_total_accuracy(self,accuracy_list):

        #totalling all accuracy
        total = accuracy_list.sum()

        length = len(accuracy_list)

        #getting final accuracy by dividing total accuracy with its length
        accuracy = total/length
        return accuracy

    def predict_service(self):

        self.test_service_input["part_pred"] = "1"

        for index, row in self.test_service_input.iterrows():

            question = row['question']

            pred_service = self.service_extractor.predict_label(question)

            self.test_service_input.at[index,'part_pred'] = pred_service

        #Dropping null and resetting index
        self.test_service_input.dropna(inplace=True)

        self.test_service_input.reset_index(drop=True, inplace=True)

        #initializing accuracy column value
        self.test_service_input["accuracy"] = " "

        #iterating over rows and finding accuracy
        for index,row in self.test_service_input.iterrows():

            y_true = row['part']

            y_pred = row['part_pred']

            accuracy = self.accuracy_score_individual(y_true,y_pred)

            self.test_service_input.at[index,'accuracy'] = accuracy

        model_accuracy = self.calculate_total_accuracy(self.test_service_input['accuracy'])

        return model_accuracy

    def save_model(self,dataframe,output_path):

        dataframe.to_csv(output_path,index=False)

        return True

    def predict_accuracy(self):

        model_accuracy = self.predict_service()

        self.save_model(self.test_service_input,self.test_service_output)

        return model_accuracy


if __name__ == "__main__":
    test_service = TestService()
    prediction_score = test_service.predict_accuracy()
    print(prediction_score)