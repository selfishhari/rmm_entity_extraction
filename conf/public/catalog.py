

#--------------------file name---------------------------
BRAND_MODEL = "brand model"

ALTERNATIVE_NAME = "alternative name"

TELEGRAM_QUESTION = "telegram question"

#----------------------input files-------------------------

EXCEL_SHEET = "/home/centos/rmm_entity_extraction/data/Inputs/Hari Mobile chatbot.xlsx"

brand_model_csv = "/home/centos/rmm_entity_extraction/data/Inputs/brand_model_2.csv"

alternative_name_csv = "/home/centos/rmm_entity_extraction/data/Inputs/alternative_name.csv"

brand_rank_csv = "/home/centos/rmm_entity_extraction/data/Inputs/brand_rank.csv"

#-------------------output files-------------------------

CSV_OUTPUT_BRAND_MODEL = "/home/centos/rmm_entity_extraction/data/Outputs/brand_model.csv"

CSV_OUTPUT_BRAND_RANK = "/home/centos/rmm_entity_extraction/data/Outputs/brand_rank.csv"

CSV_OUTPUT_ALTERNATIVE_NAME = "/home/centos/rmm_entity_extraction/data/Outputs/alternative_name.csv"

#--------------database config--------------------------

DB_HOST = "localhost"

DB_NAME = "rootvj_rmm_ai"

INTERVAL = "INTERVAL 1 YEAR" #Time Interval for extract query

SHOP_TABLE = "shops"

CUSTOMER_ENTITY_TABLE = "cs_entity"

PIN_COLUMN = "pin"

DATE_COLUMN = "date"

USER_ID_COLUMN = "user_id"

LATITUDE_COLUMN = "latitude"

LONGITUDE_COLUMN = "longitude"

#-----------test class----------------------

TEST_OUTPUT_BRAND_CSV = "/home/centos/rmm_entity_extraction/data/Outputs/test_output_brand.csv"

TEST_OUTPUT_MODEL_CSV = "/home/centos/rmm_entity_extraction/data/Outputs/test_output_model.csv"

TEST_OUTPUT_SERVICE_CSV = "/home/centos/rmm_entity_extraction/data/Outputs/test_output_service.csv"

TEST_SET_BRAND = "/home/centos/rmm_entity_extraction/data/Inputs/test_set_brands2.csv"

TEST_SET_MODELS = "/home/centos/rmm_entity_extraction/data/Inputs/test_set_models2.csv"

TEST_SET_SERVICE = "/home/centos/rmm_entity_extraction/data/Inputs/test_service.csv"

POPULATE_SET = "/home/centos/rmm_entity_extraction/data/Inputs/populate_set.csv"
#--------parameters------------

FUZZ_PERCENT = 75   #Matching percent

ITERATE_LIMIT = 20000 #Iterate limit for pincode

ITERATE_LIMIT_KMS = 1000 # Iterate limit for lat long matching

#------Trained Models-------------------------

VECTORIZED_MODEL = "/home/centos/rmm_entity_extraction/data/Models/vectorizer.model" #TFIDF vector model

NAIVE_BAYES_MODEL = "/home/centos/rmm_entity_extraction/data/Models/naivebayes.model" #Naive Bayes Model

KNOWN_CATEGORIES = ['service', 'others', 'combo', 'file', 'motherboard', 'flash box']

OTHER_CATEGORIES = ['unlock dongle','flash tool','flash box,service','repairing tool', 'battery',
 'mobile repair parts,combo,battery,cover,touch display', 'flash dongle', 'ic','back panel','country unlock sim',
 'data cable','glue remover','short remover ','microscope','mobile','oca machine','power bank']


substring =  ["combo"]
