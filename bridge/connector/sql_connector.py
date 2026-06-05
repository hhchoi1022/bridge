#%%
import psycopg2
from psycopg2 import pool, sql, errors
from astropy.table import Table
import uuid
import numpy as np
import pandas as pd
from psycopg2.extras import execute_values
from contextlib import contextmanager
import logging

from bridge.configuration import Configuration

class SQLConnector:
    def __init__(self,
                 id_user: str = None,
                 pwd_user: str = None,
                 host_user: str = None,
                 port_user: int = None,
                 db_name: str = None,
                 pool_minconn: int = None,
                 pool_maxconn: int = None):
        self.config = Configuration(config_filenames=['sqlconnector.config'])
        if id_user is None:
            id_user = self.config.id_user
        if pwd_user is None:
            pwd_user = self.config.pwd_user
        if host_user is None:
            host_user = self.config.host_user
        if port_user is None:
            port_user = self.config.port_user
        if db_name is None:
            db_name = self.config.db_name
        if pool_minconn is None:
            pool_minconn = self.config.pool_minconn
        if pool_maxconn is None:
            pool_maxconn = self.config.pool_maxconn
        self.id_user = id_user
        self.pwd_user = pwd_user
        self.host_user = host_user
        self.port_user = port_user
        self.db_name = db_name
        self.pool_minconn = pool_minconn
        self.pool_maxconn = pool_maxconn

        # PostgreSQL connection pool
        self.pool = psycopg2.pool.SimpleConnectionPool(
            minconn=self.pool_minconn,
            maxconn=self.pool_maxconn,
            user=self.id_user,
            password=self.pwd_user,
            host=self.host_user,
            port=self.port_user,
            database=self.db_name
        )

    def __repr__(self):
        return f"PostgreSQL(DB = {self.db_name}, Address = {self.id_user}@{self.host_user})"

    def connect(self):
        return self.pool.getconn()

    def disconnect(self):
        self.pool.closeall()
        print("Connection pool has been closed.")

    @contextmanager
    def get_cursor(self):
        """Context manager for database operations with automatic cleanup"""
        conn = self.connect()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            logging.error(f"Database error: {e}")
            raise
        finally:
            cursor.close()
            self.pool.putconn(conn)

    def execute(self, sql_command, params=None, commit=False):
        conn = self.connect()
        cursor = conn.cursor()
        try:
            cursor.execute(sql_command, params)
            if commit:
                conn.commit()
            return cursor
        except Exception as e:
            conn.rollback()
            print(f"Error: {e}")
            return None
        finally:
            cursor.close()
            self.pool.putconn(conn)

    # -------------------------------
    # Database & Table Utilities
    # -------------------------------
    @property
    def databases(self):
        """Get list of available databases"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute("SELECT datname FROM pg_database WHERE datistemplate = false;")
                return [db_name[0] for db_name in cursor.fetchall()]
        except Exception as e:
            logging.error(f"Error fetching databases: {e}")
            return []

    def change_db(self, db_name: str):
        self.db_name = db_name
        self.pool = psycopg2.pool.SimpleConnectionPool(
            minconn=self.pool.minconn,
            maxconn=self.pool.maxconn,
            user=self.id_user,
            password=self.pwd_user,
            host=self.host_user,
            database=self.db_name
        )

    def create_db(self, db_name: str):
        self.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)), commit=True)

    def remove_db(self, db_name: str):
        self.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(db_name)), commit=True)

    # Table utilities
    @property
    def tables(self):
        conn = self.connect()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='public';"
            )
            return [tbl_name[0] for tbl_name in cursor.fetchall()]
        except Exception as e:
            print(f"Error: {e}")
            return []
        finally:
            cursor.close()
            self.pool.putconn(conn)
            
    def create_table_from_dataframe(self, tbl_name: str, df):

        dtype_map = {
            "int64": "BIGINT",
            "float64": "DOUBLE PRECISION",
            "bool": "BOOLEAN",
            "object": "TEXT"
        }

        columns_sql = []

        for col in df.columns:
            dtype = str(df[col].dtype)
            pg_type = dtype_map.get(dtype, "TEXT")

            # Make candid primary key
            if col == "candid":
                col_def = sql.SQL("{} {} PRIMARY KEY").format(
                    sql.Identifier(col),
                    sql.SQL(pg_type)
                )
            else:
                col_def = sql.SQL("{} {}").format(
                    sql.Identifier(col),
                    sql.SQL(pg_type)
                )

            columns_sql.append(col_def)

        query = sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(
            sql.Identifier(tbl_name),
            sql.SQL(', ').join(columns_sql)
        )

        self.execute(query, commit=True)
        
    def insert_dataframe(self, tbl_name: str, df):

        df = df.where(pd.notnull(df), None)

        columns = list(df.columns)

        query = sql.SQL(
            "INSERT INTO {} ({}) VALUES %s ON CONFLICT DO NOTHING"
        ).format(
            sql.Identifier(tbl_name),
            sql.SQL(', ').join(map(sql.Identifier, columns))
        )

        values = [tuple(row) for row in df.to_numpy()]

        conn = self.connect()
        try:
            with conn.cursor() as cursor:
                execute_values(cursor, query, values)
            conn.commit()
        except Exception as e:
            conn.rollback()
            print("Insert error:", e)
        finally:
            self.pool.putconn(conn)
            
    def add_tbl(self, tbl_name: str, data: Table = None):
        if data is None:
            data = Table()
        self.execute(sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(
            sql.Identifier(tbl_name),
            sql.SQL(', ').join(map(sql.Identifier, data.columns))
        ), commit=True)

    def remove_tbl(self, tbl_name: str):
        self.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(tbl_name)), commit=True)

    def get_colnames(self, tbl_name: str):
        conn = self.connect()
        cursor = conn.cursor()
        try:
            cursor.execute(sql.SQL("SELECT column_name FROM information_schema.columns WHERE table_name = %s;"),
                          (tbl_name,))
            return [column[0] for column in cursor.fetchall()]
        except Exception as e:
            print(f"Error: {e}")
            return []
        finally:
            cursor.close()
            self.pool.putconn(conn)

    def get_column_data_types(self, tbl_name: str):
        conn = self.connect()
        cursor = conn.cursor()
        try:
            cursor.execute(sql.SQL(
                "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = %s;"),
                (tbl_name,))
            return {col[0]: col[1] for col in cursor.fetchall()}
        except Exception as e:
            print(f"Error: {e}")
            return {}
        finally:
            cursor.close()
            self.pool.putconn(conn)

    # -------------------------------
    # Data Manipulation
    # -------------------------------

    def insert_rows(self, tbl_name: str, data: Table):
        data_str = data.copy()
        for colname in data_str.columns:
            data_str[colname] = data_str[colname].astype(str)
        if 'idx' in data_str.keys():
            data_str.remove_column('idx')

        common_colnames = [col for col in data_str.colnames if col in self.get_colnames(tbl_name)]
        sql_command = sql.SQL("INSERT INTO {} ({}) VALUES %s").format(
            sql.Identifier(tbl_name),
            sql.SQL(', ').join(map(sql.Identifier, common_colnames))
        )

        values = [
            tuple(None if row[col] in ('None', '') else row[col] for col in common_colnames)
            for row in data_str
        ]

        conn = self.connect()
        try:
            with conn.cursor() as cursor:
                execute_values(cursor, sql_command, values)
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"Insert error: {e}")
        finally:
            self.pool.putconn(conn)
            
    def remove_rows(self, tbl_name: str, ids: list or str):
        if isinstance(ids, str):
            ids = [ids]
        id_list = tuple(ids)
        self.execute(sql.SQL("DELETE FROM {} WHERE trigger_id = ANY(%s)")
                     .format(sql.Identifier(tbl_name)), (id_list,), commit=True)
        
    def update_row(self, tbl_name: str, update_value, update_key, id_value, id_key='trigger_id'):
        # Normalize id_key and id_value to list
        if isinstance(id_key, str):
            id_key = [id_key]
        if isinstance(id_value, str):
            id_value = [id_value]

        # Normalize update_key and update_value to list
        if isinstance(update_key, str):
            update_key = [update_key]
            update_value = [update_value]

        # Fetch valid database columns
        valid_cols = self.get_colnames(tbl_name)

        # Filter valid update pairs
        update_pairs = [
            (key, val)
            for key, val in zip(update_key, update_value)
            if key in valid_cols
        ]

        # If nothing valid to update ? exit quietly
        if not update_pairs:
            return False

        # Build SET clause
        set_clause = sql.SQL(', ').join(
            sql.SQL("{} = %s").format(sql.Identifier(key))
            for key, _ in update_pairs
        )

        # Build WHERE clause
        where_clause = sql.SQL(' AND ').join(
            sql.SQL("{} = %s").format(sql.Identifier(key))
            for key in id_key
        )

        sql_command = sql.SQL("UPDATE {} SET {} WHERE {}").format(
            sql.Identifier(tbl_name),
            set_clause,
            where_clause
        )

        # Values for SET + WHERE
        update_vals = [val for _, val in update_pairs]
        all_values = tuple(update_vals) + tuple(id_value)

        cursor = self.execute(sql_command, all_values, commit=True)
        return cursor is not None


    # def update_row(self, tbl_name: str, update_value, update_key, id_value, id_key='trigger_id'):
    #     if isinstance(id_key, str):
    #         id_key = [id_key]
    #     if isinstance(id_value, str):
    #         id_value = [id_value]
            
            

    #     where_clause = sql.SQL(' AND ').join(
    #         sql.SQL("{} = %s").format(sql.Identifier(key)) for key in id_key
    #     )

    #     if isinstance(update_key, str):
    #         update_key = [update_key]
    #         update_value = [update_value]

    #     set_clause = sql.SQL(', ').join(
    #         sql.SQL("{} = %s").format(sql.Identifier(key)) for key in update_key
    #     )

    #     sql_command = sql.SQL("UPDATE {} SET {} WHERE {}").format(
    #         sql.Identifier(tbl_name),
    #         set_clause,
    #         where_clause
    #     )

    #     all_values = tuple(update_value) + tuple(id_value)
    #     cursor = self.execute(sql_command, all_values, commit=True)
    #     return cursor is not None

    # def get_data(self, tbl_name: str, select_key: str = '*', where_value: str = None, where_key: str = 'trigger_id', out_format: str = 'Table'):
    #     sql_command = sql.SQL("SELECT {} FROM {}").format(
    #         sql.SQL(select_key) if select_key != '*' else sql.SQL('*'),
    #         sql.Identifier(tbl_name)
    #     )
    #     params = None
    #     if where_value:
    #         sql_command = sql.SQL("SELECT {} FROM {} WHERE {} = %s").format(
    #             sql.SQL(select_key) if select_key != '*' else sql.SQL('*'),
    #             sql.Identifier(tbl_name),
    #             sql.Identifier(where_key)
    #         )
    #         params = (where_value,)

    #     conn = self.connect()
    #     cursor = conn.cursor()
    #     try:
    #         cursor.execute(sql_command, params)
    #         output = cursor.fetchall()
    #     except Exception as e:
    #         print(f"Error: {e}")
    #         output = []
    #     finally:
    #         cursor.close()
    #         self.pool.putconn(conn)

    #     keys = self.get_colnames(tbl_name) if select_key == '*' else select_key.split(',')
    #     result = Table() if out_format.lower() == 'table' else {}
    #     if len(keys) == 1:
    #         result[keys[0]] = [out[0] for out in output]
    #     else:
    #         for i, key in enumerate(keys):
    #             result[key] = [out[i] for out in output]
    #     return result

    def get_data(self, tbl_name: str, select_key: str = '*',
                where_value: str = None, where_key: str = 'trigger_id',
                out_format: str = 'Table'):

        sql_command = sql.SQL("SELECT {} FROM {}").format(
            sql.SQL(select_key) if select_key != '*' else sql.SQL('*'),
            sql.Identifier(tbl_name)
        )
        params = None
        if where_value:
            sql_command = sql.SQL("SELECT {} FROM {} WHERE {} = %s").format(
                sql.SQL(select_key) if select_key != '*' else sql.SQL('*'),
                sql.Identifier(tbl_name),
                sql.Identifier(where_key)
            )
            params = (where_value,)

        conn = self.connect()
        cursor = conn.cursor()
        try:
            cursor.execute(sql_command, params)
            output = cursor.fetchall()
            # 실행 결과의 실제 컬럼 순서를 반영
            keys = [desc[0] for desc in cursor.description]
        except Exception as e:
            print(f"Error: {e}")
            output = []
            keys = []
        finally:
            cursor.close()
            self.pool.putconn(conn)

        # 결과 포맷
        result = Table() if out_format.lower() == 'table' else {}
        if output:
            for i, key in enumerate(keys):
                result[key] = [out[i] for out in output]

        return result

    def set_data_id(self, tbl_name: str, update_all: bool = False):
        values_all = self.get_data(tbl_name=tbl_name, select_key='id,idx')
        values_to_update = values_all if update_all else values_all[values_all['id'] == None]
        uuidlist = [uuid.uuid4().hex for _ in range(len(values_to_update))]

        for id_, index in zip(uuidlist, values_to_update['idx']):
            self.update_row(tbl_name=tbl_name, update_value=id_, update_key='id', id_value=str(index), id_key='idx')

    def pool_status(self):
        print(f"Connections: min={self.pool.minconn}, max={self.pool.maxconn}")


# %%
if __name__ == "__main__":
    connector = SQLConnector()
    # Test all functions
    connector.change_db('hhchoi1022')
    print(connector.databases)
    print(connector.tables)
    print(connector.get_colnames('too_list'))
    print(connector.get_column_data_types('too_status'))
    print(connector.get_data(tbl_name = 'too_status'))
    from astropy.table import Table
    from astropy.time import Time
    tbl_insert = Table()
    #tbl_insert['trigger_id'] = [1, 2, 3]
    tbl_insert['status'] = [False, True, False]
    tbl_insert['trigger_time'] = [Time.now().iso, Time.now().iso, Time.now().iso]
    tbl_insert['multi_target'] = [False, True, False]
    tbl_insert['cent_ra'] = [1, 2, 3]
    tbl_insert['cent_dec'] = [1, 2, 3]
    tbl_insert['alert_time'] = [Time.now().iso, Time.now().iso, Time.now().iso]
    tbl_insert['is_triggered'] = [False, True, False]
    tbl_insert['objname'] = [1, 2, 3]
    tbl_insert['note'] = [1, 2, 3]
    connector.insert_rows(tbl_name = 'too_list', data = tbl_insert)
    print(connector.get_data(tbl_name = 'too_status'))
#%%
if __name__ == "__main__":
    connector = SQLConnector(
        id_user = 'gwuteam',
        pwd_user = 'kilo^N0va',
        host_user = 'proton.snu.ac.kr',
        port_user = 5433,
        db_name = 'gwu'
    )
    print(connector.tables)
    #print(connector.get_data(tbl_name = 'too_status'))
#%%
from pprint import pprint
if __name__ == "__main__":
    connector.get_colnames('survey_scienceframe')
#%%

if __name__ == "__main__":
    select_key_list = [
    'filename',
    'seeing',
    'skysig',
    'skyval',
    'ul5',
    'zp',
    'id',
    'astro_rms',
    'ra_center',
    'dec_center',
    'ellip',
    'elong',
    'jd',
    'master_bias_id',
    'master_dark_id',
    'master_flat_id',
    'mjd',
    'phot_nstars',
    'saturation_level']
    data = connector.get_data('survey_processedscienceframe', select_key = ','.join(select_key_list))
    pprint(data)
# %%
if __name__ == "__main__":
    select_key_list = [
        'object_name',
        'obstime',
        'exptime',
        'object_ra',
        'object_dec',
        'object_type',
        'original_filename',
        'filename_metadata',
        'obsmode',
        'specmode',
        'filter_id',
    ]
    data = connector.get_data('survey_scienceframe', select_key = ','.join(select_key_list))

# %%
if __name__ == "__main__":
    connector.get_data('facility_filterposition')
# %%
