import psycopg2
import pandas as pd
from src.config import config
import datetime as datetime
import time

# returns a given SQL SELECT query to a pandas DataFrame object
def query_to_df(query= "SELECT * FROM power_weather LIMIT 15"):
    
    params=config()

    try:
        conn = psycopg2.connect(**params)
        cursor = conn.cursor()
        
        cursor.execute(query)
        df = pd.DataFrame(cursor.fetchall())
        print(cursor.description)
        df.columns = [i[0] for i in cursor.description]
        #df.set_index('id', drop=True, inplace=True)
        print('df Created')

    except (Exception, psycopg2.Error) as error :
        print ("Error while connecting to PostgreSQL:", error)

    if(conn):
        cursor.close()
        conn.close()
    return df

#executes a given SQL query in the postgresql database
def execute_query(query):
    
    params=config()

    try:
        conn = psycopg2.connect(**params)
        cursor = conn.cursor()
        
        cursor.execute(query)
        print("Query Executed")
        
    except (Exception, psycopg2.Error) as error :
        print ("Error while connecting to PostgreSQL:", error)

    if(conn):
        conn.commit()
        cursor.close()
        conn.close()

#creates a new database from given df
def df_to_sql(df, name):
    
    params=config()

    try:
        engine = create_engine(**params)
        df.to_sql(name, engine)
        
        print("Database Created")
        
    except (Exception, psycopg2.Error) as error :
        print ("Error while connecting to PostgreSQL:", error)


def update_sql_from_df(df, name):
    
    params=config()
    
    try:
        conn = psycopg2.connect(**params)
        cursor = conn.cursor()
        
        for index, row in df.iterrows():
            update_query = f"INSERT INTO {name} VALUES ({row.game_id}, {row.name}, {row.p1}, {row.p2}, {row.winner}, {row.model_choice}, {row.model0}, {row.model1}, {row.model2}, {row.model3}, {row.model4}, {row.model5}, '{row.timestamp if type(row.timestamp) != int else datetime.datetime.fromtimestamp(int(str(row.timestamp)[:10]))}', '{row.ip_address}')"
            cursor.execute(update_query, name)
        print("Database Updated")
        
    except (Exception, psycopg2.Error) as error :
        print ("Error while connecting to PostgreSQL:", error)

    finally:
        if(conn):
            conn.commit()
            cursor.close()
            conn.close()