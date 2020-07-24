from test_class import AccuracyMeasure
from created import MatchUser
from update_brands import UpdateBrandModel
from update_alternative_name import UpdateAlternativeName
from update_brand_rank import UpdateBrandRank
import pandas as pd

def run():
    test_set_brand = pd.read_csv("/home/viky/Desktop/freelancer/FLASK/app/data/input/test_set_brands2.csv")
    test_set_model = pd.read_csv("/home/viky/Desktop/freelancer/FLASK/app/data/input/test_set_models2.csv")
    accuracy_measure = AccuracyMeasure(test_set_brand,test_set_model)
    mod, brand = accuracy_measure.calculate_accuracy()
    return mod,brand

def brand():
    update_brand = UpdateBrandModel()
    update_brand.update()

def alt_name():
    update_alternative = UpdateAlternativeName()
    update_alternative.update()

def brand_rank():
    brand_rank = UpdateBrandRank()
    brand_rank.update()

if __name__ == "__main__":
    #brand = brand()
    #alternative = alt_name()
    brand_rank = brand_rank()
    # mod,brand = run()
    # print(mod*100)
    # print(brand*100)
