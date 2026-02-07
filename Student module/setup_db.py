import mysql.connector
import pandas as pd
#establishing the connection
conn = mysql.connector.connect(
   user='root', password='root', host='localhost', database='itemresponsetheory'
)

#Creating a cursor object using the cursor() method
cursor = conn.cursor()

#Dropping EMPLOYEE table if already exists.
cursor.execute("DROP TABLE IF EXISTS ITEM")

#Creating table as per requirement
sql ='''CREATE TABLE ITEM(
ID int NOT NULL AUTO_INCREMENT,
a FLOAT not null,
b FLOAT not null,
c FLOAT not null,
d VARCHAR(2500) not null,
answer varchar(2500) not null,
PRIMARY KEY (ID)
)'''
cursor.execute(sql)
cursor.execute("DROP TABLE IF EXISTS TESTS")
#Closing the connection
sql ='''CREATE TABLE TESTS(
ID int NOT NULL AUTO_INCREMENT,
TESTID VARCHAR(250) NOT NULL,
ItemID int NOT NULL,
code VARCHAR(250) not null,
primary key (id)
)'''



df = pd.read_csv('items.csv')
li = df.values.tolist()

cursor.execute(sql)
for k in li:
    cursor.execute("insert into item(a,b,c,d,answer) values(%s,%s,%s,%s,%s)",(k[0],k[1],k[2],k[3],k[4]))
conn.commit()

df = pd.read_csv('tests.csv')
li = df.values.tolist()
for k in li:
    cursor.execute("insert into tests(TESTID,ItemID,code) values(%s,%s,%s)",(k[0],k[1],k[2]))
conn.commit()

conn.close()
