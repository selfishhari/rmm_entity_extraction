

#--------------------file name---------------------------
BRAND_MODEL = "brand model"

ALTERNATIVE_NAME = "alternative name"

TELEGRAM_QUESTION = "telegram question"

#----------------------input files-------------------------

EXCEL_SHEET = "/home/ec2-user/rmm_entity_extraction/data/03_model_inputs/Hari Mobile chatbot.xlsx"

brand_model_csv = "/home/ec2-user/rmm_entity_extraction/data/03_model_inputs/brand_model_2.csv"

alternative_name_csv = "/home/ec2-user/rmm_entity_extraction/data/03_model_inputs/alternative_name.csv"

brand_rank_csv = "/home/ec2-user/rmm_entity_extraction/data/03_model_inputs/brand_rank.csv"

#-------------------output files-------------------------

CSV_OUTPUT_BRAND_MODEL = "/home/ec2-user/rmm_entity_extraction/data/05_output/brand_model.csv"

CSV_OUTPUT_BRAND_RANK = "/home/ec2-user/rmm_entity_extraction/data/05_output/brand_rank.csv"

CSV_OUTPUT_ALTERNATIVE_NAME = "/home/ec2-user/rmm_entity_extraction/data/05_output/alternative_name.csv"

#--------------database config--------------------------

DB_HOST = "localhost"

DB_NAME = "rmm_entity"

INTERVAL = "INTERVAL 1 DAY"

SHOP_TABLE = "shops"

CUSTOMER_ENTITY_TABLE = "cs_entity"

PIN_COLUMN = "pin"

DATE_COLUMN = "date"

USER_ID_COLUMN = "user_id"

LATITUDE_COLUMN = "latitude"

LONGITUDE_COLUMN = "longitude"

#-----------test class----------------------

TEST_OUTPUT_BRAND_CSV = "/home/ec2-user/rmm_entity_extraction/src/app/data/output/test_output_brand.csv"

TEST_OUTPUT_MODEL_CSV = "/home/ec2-user/rmm_entity_extraction/src/app/data/output/public/test_output_model.csv"

TEST_SET_BRAND = "/home/ec2-user/rmm_entity_extraction/src/app/data/input/test_set_brands2.csv"

TEST_SET_MODELS = "/home/ec2-user/rmm_entity_extraction/src/app/data/input/test_set_models2.csv"

#--------parameters------------

FUZZ_PERCENT = 75

ITERATE_LIMIT = 20000