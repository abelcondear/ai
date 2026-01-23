import sqlalchemy
import sys

from sqlalchemy import create_engine, MetaData, Table, Column, Numeric, Integer, VARCHAR
from sqlalchemy.engine import result
from sqlalchemy import select
from sqlalchemy import text
from sqlalchemy.dialects.mssql.base import ischema_names
from sqlalchemy.types import String

engine = create_engine(
   "mssql+pyodbc://localhost/master"
   "?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server",
    isolation_level="AUTOCOMMIT"
)

class SysName(String):
   __visit_name__ = "sysname"

ischema_names["sysname"] = SysName

# initialize the Metadata Object
meta = MetaData()

# get snapshot from database
meta.reflect(engine) 

tablename = 'messages'

if not sqlalchemy.inspect(engine).has_table("messages"):
   # create a table schema
   messages = Table(
      tablename, meta,
      Column('id', Integer, primary_key=True),
      Column('message', VARCHAR),
      Column('is_priority', Numeric)
   )

   # create table if it does not exist yet
   meta.create_all(engine) 

   # get second snapshot from database
   meta.reflect(engine) 
   
   print(f"Table \"{tablename}\" was created.")
else:
   # get table schema
   messages = Table(
      'messages', meta   
   )      

id_value = int(sys.argv[1:][0])
message_value = sys.argv[1:][1]
priority_value = int(sys.argv[1:][2])

# insert records into the table
statement = messages.insert().values(id=id_value, 
                                 message=message_value,
                                 is_priority=priority_value)

with engine.connect() as conn:
   # execute the inserts
   conn.execute(statement)   
   print("New record was inserted.")

# clean up sql alchemy cache
engine.dispose() 
