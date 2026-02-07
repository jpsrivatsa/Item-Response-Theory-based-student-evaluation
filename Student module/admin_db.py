import mysql.connector
import pandas as pd
#establishing the connection
conn = mysql.connector.connect(
   user='root', password='root', host='localhost', database='irt'
)

cursor = conn.cursor()
cursor.execute("DROP TABLE IF EXISTS admin")
#Closing the connection
sql ='''CREATE TABLE admin(
ID int NOT NULL AUTO_INCREMENT,
username VARCHAR(250) NOT NULL,
password VARCHAR(250) not null,
primary key (id)
)'''
cursor.execute("insert into admin(username,password) values(%s,%s)",("admin","admin"))
cursor.execute(sql)
conn.commit()
conn.close()