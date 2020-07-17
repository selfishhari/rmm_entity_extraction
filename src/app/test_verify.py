from test_class import AccuracyMeasure
from created import MatchUser
from update_brands import UpdateBrandModel
import pandas as pd

def run():
    test_set_brand = pd.read_csv("/home/ec2-user/rmm_entity_extraction/src/app/conf/test_set_brands.csv")
    test_set_model = pd.read_csv("/home/ec2-user/rmm_entity_extraction/src/app/conf/test_set_models.csv")
    accuracy_measure = AccuracyMeasure(test_set_brand,test_set_model)
    mod, brand = accuracy_measure.calculate_accuracy()
    return mod,brand
def brand():
    update_brand = UpdateBrandModel()
    update_brand.update()

if __name__ == "__main__":
    #brand = brand()
    mod,brand = run()
    print(mod*100)
    print(brand*100)