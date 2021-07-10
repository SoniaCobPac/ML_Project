import pymysql

class MySQL:

    def __init__(self, IP_DNS, USER, PASSWORD, BD_NAME, PORT):
        self.IP_DNS = IP_DNS
        self.USER = USER
        self.PASSWORD = PASSWORD
        self.BD_NAME = BD_NAME
        self.PORT = PORT
        self.SQL_ALCHEMY = 'mysql+pymysql://' + self.USER + ':' + self.PASSWORD + '@' + self.IP_DNS + ':' + str(self.PORT) + '/' + self.BD_NAME
        # 'mysql+pymysql://user:password@91.76.54.33:20001/apr_july_2021_tb'
    def connect(self):
        # Open database connection
        self.db = pymysql.connect(host=self.IP_DNS,
                                  user=self.USER, 
                                  password=self.PASSWORD, 
                                  database=self.BD_NAME, 
                                  port=self.PORT)
        # prepare a cursor object using cursor() method
        self.cursor = self.db.cursor()
        print("Connected to MySQL server [" + self.BD_NAME + "]")
        return self.db

    def close(self):
        # disconnect from server
        self.db.close()
        print("Close connection with MySQL server [" + self.BD_NAME + "]")
    
    def execute_interactive_sql(self, sql, delete=False):
        """ NO SELECT """
        result = 0
        try:
            # Execute the SQL command
            self.cursor.execute(sql)
            # Commit your changes in the database
            self.db.commit()
            print("Executed \n\n" + str(sql) + "\n\n successfully")
            result = 1
        except Exception as error:
            print(error)
            # Rollback in case there is any error
            self.db.rollback()
        return result
        
    def execute_get_sql(self, sql):
        """SELECT"""
        results = None
        print("Executing:\n", sql)
        try:
            # Execute the SQL command
            self.cursor.execute(sql)
            # Fetch all the rows in a list of lists.
            results = self.cursor.fetchall()
        except Exception as error:
            print(error)
            print ("Error: unable to fetch data")
        
        return results

    def generate_insert_into_music_sql(self, to_insert):
        """
        """
        piece = to_insert[0]
        note = to_insert[1]

        sql = """INSERT INTO df_music
            (PIECE, NOTE)
            VALUES
            (NOW(), '""" + piece + """', '""" + note + """')"""

        sql = sql.replace("\n", "").replace("            ", " ")
        return sql

    def generate_insert_results_sql(self, to_insert):
        """
        """
        models = to_insert[0]
        params = to_insert[1]
        loss = to_insert[2]
        rmse = to_insert[3]
        accuracy = to_insert[4]

        sql = """INSERT INTO df_results
            (MODEL, PARAMETERS, LOSS, RMSE, ACCURACY)
            VALUES
            (NOW(), '""" + models + """', '""" + params + """', '""" + loss + """', '""" + rmse + """', '""" +accuracy + """')"""

        sql = sql.replace("\n", "").replace("            ", " ")
        return sql

