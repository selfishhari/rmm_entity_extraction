import sys
sys.path.append("/home/ec2-user/rmm_entity_extraction")

from src.test_class.test_class import AccuracyMeasure
import pandas as pd

from conf.public import catalog


def run():
    catalog_dict = vars(catalog)

    test_set_brand = pd.read_csv(catalog_dict['TEST_SET_BRAND'])

    test_set_model = pd.read_csv(catalog_dict['TEST_SET_MODELS'])

    accuracy_measure = AccuracyMeasure(test_set_brand,test_set_model)

    mod, brand = accuracy_measure.calculate_accuracy()

    return mod,brand


if __name__ == "__main__":

    mod,brand = run()

    print(mod*100)

    print(brand*100)
