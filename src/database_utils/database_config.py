import sqlalchemy



class DataBaseEngine():

    def __init__(self,catalog,credentials):

        self.catalog = catalog

        self.credentials = credentials

        self.user_name = self.credentials['DB_USERNAME']

        self.password = self.credentials['DB_PASSWORD']

        self.host = self.catalog['DB_HOST']

        self.db_name =self.catalog['DB_NAME']

    def config_engine(self):

        db_engine = sqlalchemy.create_engine(
                'mysql+pymysql://{0}:{1}@{2}/{3}?charset=utf8mb4'.format(self.user_name,
                                                self.password,
                                                self.host,
                                                self.db_name))

        return db_engine