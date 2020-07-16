from test_class import AccuracyMeasure
from created import MatchUser
import pandas as pd

def run():
    test_set = pd.read_csv("/home/viky/Desktop/freelancer/Test Set.csv")
    accuracy_measure = AccuracyMeasure(test_set)
    mod, brand = accuracy_measure.calculate_accuracy()
    return mod,brand


if __name__ == "__main__":
    mod,brand = run()
    print(mod*100)
    print(brand*100)