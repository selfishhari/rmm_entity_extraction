# rmm_entity_extraction
Entity Extraction Sytem for RMM's customer support interface

## Installation
- Install Python 3.6+ version 
- Install virtual environment
- Install MySQL
- Create a virtual environment. 
- Activate the virtual environment and install the dependency packages using requirements.txt file
- Start the service using the command "python deploy_extraction_service.py"

## Table of contents
- deploy_extraction_service
- entity_extractor
- service_extractor
- fuzz_match
- update_brands
- update_alternative_names
- update_brand_ranks
- test_class
- database_config


### deploy_extraction_service
The file consists of api's for customer support interface

### entity_extractor
The entity extractor is used to extract brand model and service from given text and get a suitable match from the database. 

### service_extractor
The service extractor is used to extract service using trained Naive Bayes Model.

### fuzz_match
The fuzz_match module is used to get appropriate match for fuzzy spellings. 

### update_brands
The update brands module is used to extract brand and model names from telegram text and update the existing brand and model excel file

### update_alternative_names
The update brands module is used to extract alternative names of brand using fuzz_match from telegram text and update the existing alternative names excel file.

### update_brand_ranks
The update brand ranks is used to update the ranks of brands according to their frequency of occurance in the telegram text.

### test_class
The test class is used to find the accuracy of entity_extractor.

### database_config
The database config is used the configure the database and helps the entity_extractor to extract matching users from the database
