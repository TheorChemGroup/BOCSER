import sqlite3
import os
from typing import List

class Connector:
    
    def __init__(
        self
    ) -> None:
        pass

    def set_request(
        self,
        request : str
    ) -> None:
        pass

    def get_request(
        self,
        request : str
    ) -> List:
        pass

class LocalConnector(Connector):

    def __init__(
        self, 
        db_filename : str = 'dihedral_logs.db'
    ) -> None:
        self.db_filename = db_filename
        if not os.path.isfile(db_filename):
            print("No database file located!")
            raise FileNotFoundError(db_filename)

    def set_request(
        self,
        request : str
    ) -> None:
        try:
            connection = sqlite3.connect(self.db_filename)
            cursor = connection.cursor()
            cursor.execute(request)
            connection.commit()
            connection.close()
        except Exception as e:
            print("Something went wrong with db")
            raise e
    
    def get_request(
        self,
        request : str
    ) -> List:
        try:
            connection = sqlite3.connect(self.db_filename)
            cursor = connection.cursor()
            result = cursor.execute(request).fetchall()
            connection.commit()
            connection.close()
            return result
        except Exception as e:
            print("Something went wrong with db")
            raise e

