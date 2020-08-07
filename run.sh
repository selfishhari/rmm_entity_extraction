
#!/bin/bash
export rmm_user="$rmm_user"
export rmm_password="$rmm_password"
source /home/centos/rmm_entity_extraction/rmm_venv/bin/activate            #change path for virtual environment
cd /home/centos/rmm_entity_extraction					   #change path for working directory
gunicorn -c gunicorn.conf -b 0.0.0.0:5010 deploy_extraction_service:app


