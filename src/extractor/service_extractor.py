import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from stop_words import get_stop_words
from fuzzywuzzy import fuzz


class ServiceExtractor():

    def __init__(self,catalog):

        self.catalog = catalog

        self.vectorizer = joblib.load(self.catalog['VECTORIZED_MODEL'])

        self.naivebayes = joblib.load(self.catalog['NAIVE_BAYES_MODEL'])

        self.known_categories = self.catalog['KNOWN_CATEGORIES']

        self.other_categories = self.catalog['OTHER_CATEGORIES']


    def _check(self,string, sub_str):

        if (str(string).find(str(sub_str)) == -1):

            return False

        else:

            return True

    def _decode_target_var(self,target_array):

        decoded_target = [self.known_categories[x] for x in target_array]

        return decoded_target

    def string_sub_search(self,string):
        """Checking the string for possible matches in known categories"""

        all_categories = self.known_categories + self.other_categories

        all_categories.remove('others')

        service = []

        for x in all_categories:

            if self._check(string,x):

                service.append(x)
        return service

    def _search_other_categories(self,string):
        """While the prediction is 'others', checking for sub category using fuzz match"""

        others_category = None

        max_percent = 0

        for category in self.other_categories:

            fuzz_percent = fuzz.token_set_ratio(category,string)

            if fuzz_percent > 70 and fuzz_percent > max_percent:

                others_category = category


        if others_category is not None:

            service = others_category

        else:

            service = "others"

        return service

    def predict_label(self,string):

        """Predicting the service from the given text"""

        #First stage of search with known categories
        string_sub_search_result = self.string_sub_search(string)

        #if there is no match in first stage, then predicting the service using naivebayes
        if len(string_sub_search_result)<1:

            #vectorizing the string
            vectorized_string = self.vectorizer.transform([string])

            #predicting the label
            preds_label = self.naivebayes.predict(vectorized_string.toarray())

            #decoding the label
            preds_label_dec = self._decode_target_var(preds_label)

            service = preds_label_dec[0]

            #if the predicted label is others, then searching for other categories
            if service == "others":

                service = self._search_other_categories(string)


        else:
            service = string_sub_search_result[0]

        return service