from fuzzywuzzy import fuzz

# spellcheck main class
class SpellCheck:

    # initialization method
    def __init__(self, word_list):
        # open the dictionary file
        self.data = word_list

        # change all the words to lowercase
        data = [i.lower() for i in self.data]

        # remove all the duplicates in the list
        data = set(data)

        # store all the words into a class variable dictionary
        self.dictionary = list(data)

    # string setter method
    def check(self, string_to_check):
        # store the string to be checked in a class variable
        self.string_to_check = string_to_check

    # this method returns the possible suggestions of the correct words
    def suggestions(self):
        # store the words of the string to be checked in a list by using a split function
        string_words = self.string_to_check.split()

        # a list to store all the possible suggestions
        suggestions = []

        # loop over the number of words in the string to be checked
        for i in range(len(string_words)):

            # loop over words in the dictionary
            for name in self.dictionary:

                # if the fuzzywuzzy returns the matched value greater than 80
                if fuzz.ratio(string_words[i].lower(), name.lower()) >= 75:

                    if string_words[i].lower() != name.lower():

                        # checking for mi and redmi to change to xiaomi
                        if name =="redmi" or name == "mi":

                            name = "xiaomi"

                        # append the dict word to the suggestion list
                        suggestions.append({"brand":name.lower(),"alt":string_words[i].lower()})



        # return the suggestions list
        return suggestions


    # this method returns the corrected string of the given input
    def correct(self):
        # store the words of the string to be checked in a list by using a split function
        string_words = self.string_to_check.split()

        # loop over the number of words in the string to be checked
        for i in range(len(string_words)):

            # initiaze a maximum probability variable to 0
            max_percent = 0

            # loop over the words in the dictionary
            for name in self.dictionary:

                # calulcate the match probability
                percent = fuzz.ratio(string_words[i].lower(), name.lower())

                # if the fuzzywuzzy returns the matched value greater than 80
                if percent >= 75:

                    # if the matched probability is
                    if percent > max_percent:

                        # change the original value with the corrected matched value
                        string_words[i] = name

                    # change the max percent to the current matched percent
                    max_percent = percent

        # return the cprrected string
        return " ".join(string_words)
