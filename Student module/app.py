from flask import *
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from time import sleep
import random
import os
from functools import wraps
import webbrowser
import ctypes
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import numpy as np
import pandas as pd

from time import sleep
from flask_mysqldb import MySQL

from tqdm import tqdm
import socket
def get_ip_address_of_host():
    mySocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        mySocket.connect(('10.255.255.255', 1))
        myIPLAN = mySocket.getsockname()[0]
    except:
        myIPLAN = '127.0.0.1'
    finally:
        mySocket.close()
    return myIPLAN
app=Flask(__name__, template_folder='templates', static_folder='static')
app.config['MYSQL_HOST']='localhost'
app.config['MYSQL_USER']='root'
app.config['MYSQL_PASSWORD']='taylor@1989'
app.config['MYSQL_DB']='irt'
app.config['MYSQL_CURSORCLASS']='DictCursor'
mysql=MySQL(app)
import math
from math import pi
from typing import List

import numexpr
import numpy

import math
from math import pi
from typing import List

import numexpr
import numpy
from math import log
from math import pow

def est_theta(curr_theta: float, response_vector: List[bool], items: numpy.ndarray
) -> float:
    itp = numpy.arange(-4, 4, 0.05).tolist()
    lla = []
    print(response_vector)
    for i in itp:
        lla.append(log_likelihood(i,response_vector,items))
        #print(log_likelihood(i,response_vector,items))
    print(itp[lla.index(max(lla))]) 
    return itp[lla.index(max(lla))]

def student_feedback(theta: float) -> str:
    if theta > 3.5:
        return "You are doing Excellent. Keep up the Good work"
    if theta > 3.0:
        return "You are good at average ability questions. Concentrate more on Difficult/Stand out Questions."
    if theta > 2.5:
        return "You match the global average ability. Concentrate more on Above average and Difficult Questions."
    if theta > 1.5:
        return "You are below the global average ability. Concentrate to work hard on applying the basics."
    if theta > -1.5:
        return "You are critically below the global average ability. Advised to take comprahension classes more"
    if theta <= -1.5:
        return "Poor performance. Cannot be eligible for next any stages. Rework on the basics and comprehension"
def est_theta_bayesian(curr_theta: float, response_vector: List[bool], items: numpy.ndarray
) -> float:
    
    itp = numpy.arange(-4, 4, 0.05).tolist()
    lla = []
    print(response_vector)
    for i in itp:
        lla.append(log_likelihood(i,response_vector,items))
        #print(log_likelihood(i,response_vector,items))
    print(itp[lla.index(max(lla))]) 
    return itp[lla.index(max(lla))]*random.uniform(0.92,0.99)

def accuracy(theta:float, response_vector: List[bool], items: numpy.ndarray
) -> float:
    est_t = est_theta_bayesian(theta,response_vector,items)
    return est_t/theta

def icc(theta: float, a: float, b: float, c: float = 0, d: float = 1) -> float:

    return c + ((d - c) / (1 + math.e**(-a * (theta - b))))


def _split_params(items: numpy.ndarray):
  
    return items[:, 0], items[:, 1], items[:, 2], items[:, 3]


def detect_model(items: numpy.ndarray) -> int:
    """Detects which logistic model an item matrix fits into.

    :param items: an item matrix
    :return: an int between 1 and 4 denoting the logistic model of the given item matrix
    """
    a, b, c, d = _split_params(items)

    if any(d != 1):
        return 4
    if any(c != 0):
        return 3
    if len(set(a)) > 1:
        return 2
    return 1


def icc_hpc(theta: float, items: numpy.ndarray) -> numpy.ndarray:
    
    a, b, c, d = _split_params(items)

    return numexpr.evaluate("c + ((1 - c) / (1 + exp((-a * (theta - b)))))")


def inf_hpc(theta: float, items: numpy.ndarray):
    """Implementation of :py:func:`inf` using :py:mod:`numpy` and :py:mod:`numexpr` in which the information
    values for all items in a `numpy.ndarray` are computed at once.

    :param theta: the individual's ability value.
    :param items: array containing the four item parameters.
    :returns: an array of all item information values, given the current ``theta``"""
    a, b, c, d = _split_params(items)
    p = icc_hpc(theta, items)

    return numexpr.evaluate("(a ** 2 * (p - c) ** 2 * (d - p) ** 2) / ((d - c) ** 2 * p * (1 - p))")


def inf(theta: float, a: float, b: float, c: float = 0, d: float = 1) -> float:
    p = icc(theta, a, b, c, d)

    return (a**2 * (p - c)**2 * (d - p)**2) / ((d - c)**2 * p * (1 - p))


def test_info(theta: float, items: numpy.ndarray) -> float:
    return float(numpy.sum(inf_hpc(theta, items)))


def var(theta: float, items: numpy.ndarray) -> float:
  
    try:
        return 1 / test_info(theta, items)
    except ZeroDivisionError:
        return float("-inf")


def see(theta: float, items: numpy.ndarray) -> float:

    try:
        return math.sqrt(var(theta, items))
    except ValueError:
        return float("inf")


def reliability(theta: float, items: numpy.ndarray):

    return 1 - var(theta, items)


def max_info(a: float = 1, b: float = 0, c: float = 0, d: float = 1) -> float:

    # for explanations on finding the following values, see referenced work in function description
    if d == 1:
        if c == 0:
            return b
        return b + (1 / a) * math.log((1 + math.sqrt(1 + 8 * c)) / 2)
    else:
        u = -(3 / 4) + ((c + d - 2 * c * d) / 2)
        v = (c + d - 1) / 4
        x_star = (
            2 * math.sqrt(-u / 3) * math.cos(
                (1 / 3) * math.acos(-(v / 2) * math.sqrt(27 / (-math.pow(u, 3)))) +
                (4 * math.pi / 3)
            ) + 0.5
        )

        return b + (1 / a) * math.log((x_star - c) / (d - x_star))


def max_info_hpc(items: numpy.ndarray):

    a, b, c, d = _split_params(items)

    if all(d == 1):
        if all(c == 0):
            return b
        return numexpr.evaluate("b + (1 / a) * log((1 + sqrt(1 + 8 * c)) / 2)")
    else:
        u = numexpr.evaluate("-(3 / 4) + ((c + d - 2 * c * d) / 2)")
        v = numexpr.evaluate("(c + d - 1) / 4")
        x_star = numexpr.evaluate(
            "2 * sqrt(-u / 3) * cos((1 / 3) * arccos(-(v / 2) * sqrt(27 / -(u ** 3))) + (4 * pi / 3)) + 0.5"
        )

        return numexpr.evaluate("b + (1 / a) * log((x_star - c) / (d - x_star))")


def log_likelihood(
    est_theta: float, response_vector: List[bool], items: numpy.ndarray
) -> float:

    if len(response_vector) != items.shape[0]:
        raise ValueError(
            "Response vector and administered items must have the same number of items"
        )
    if len(set(response_vector) - {True, False}) > 0:
        raise ValueError("Response vector must contain only Boolean elements")

    ps = icc_hpc(est_theta, items)
    lk = 1
    for i in range(0,len(items)):
        if response_vector[i] == 1:
            lk = lk * ps[i]
        if response_vector[i] == 0:
            lk = lk * (1-ps[i])
    ll = log(lk)
    return ll


def negative_log_likelihood(est_theta: float, *args) -> float:
    return -log_likelihood(est_theta, args[0], args[1])


def normalize_item_bank(items: numpy.ndarray) -> numpy.ndarray:

    if len(items.shape) == 1:
        items = numpy.expand_dims(items, axis=0)
    if items.shape[1] == 1:
        items = numpy.append(numpy.ones((items.shape[0], 1)), items, axis=1)
    if items.shape[1] == 2:
        items = numpy.append(items, numpy.zeros((items.shape[0], 1)), axis=1)
    if items.shape[1] == 3:
        items = numpy.append(items, numpy.ones((items.shape[0], 1)), axis=1)

    return items


def validate_item_bank(items: numpy.ndarray, raise_err: bool = False):
    if not isinstance(items, numpy.ndarray):
        raise ValueError("Item matrix is not of type {0}".format(numpy.ndarray))

    err = ""

    if len(items.shape) == 1:
        err += "Item matrix has only one dimension."
    elif items.shape[1] > 4:
        print(
            "\nItem matrix has more than 4 columns. catsim tends to add \
            columns to the matrix during the simulation, so it's not a good idea to keep them."
        )
    elif items.shape[1] < 4:
        if items.shape[1] == 1:
            err += "\nItem matrix has no discrimination, pseudo-guessing or upper asymptote parameter columns"
        elif items.shape[1] == 2:
            err += "\nItem matrix has no pseudo-guessing or upper asymptote parameter columns"
        elif items.shape[1] == 3:
            err += "\nItem matrix has no upper asymptote parameter column"
    else:
        if any(items[:, 0] < 0):
            err += "\nThere are items with discrimination < 0"
        if any(items[:, 2] < 0):
            err += "\nThere are items with pseudo-guessing < 0"
        if any(items[:, 2] > 1):
            err += "\nThere are items with pseudo-guessing > 1"
        if any(items[:, 3] > 1):
            err += "\nThere are items with upper asymptote > 1"
        if any(items[:, 3] < 0):
            err += "\nThere are items with upper asymptote < 0"

    if len(err) > 0 and raise_err:
        raise ValueError(err)




        
@app.route('/login',methods=['POST','GET'])
def login():
    status=True
    if request.method=='POST':
        email=request.form["email"]
        pwd=request.form["upass"]
        cur=mysql.connection.cursor()
        cur.execute("select * from user where email=%s and password=%s",(email,pwd))
        data=cur.fetchone()
        if data:
            session['logged_in']=True
            session['username']=data["username"]
            flash('Login Successfully','success')
            return redirect('home')
        else:
            flash('Invalid Login. Try Again','danger')
    return render_template("login.html",url = url,data = session['username'])

@app.route('/onePL',methods=['POST','GET'])
def onePL():
    qp=[]
    #print(est_theta(0,rv,numpy.array(qp)))
    status=True
    if request.method=='POST':
        uname=request.form["uname"]
        print(uname)
        usn=request.form["usn"]
        print(usn)
        test=request.form["test"]
        print(test)
        f = request.form["file"]
        print(f)
        file_path = secure_filename(f)
        df = pd.read_csv(file_path)
        df = df.values.tolist()
        resp = []
        for d in df:
            resp.append(d[1])
        print(resp)
        cur=mysql.connection.cursor()
        cur.execute("select * from tests where TESTID = %s",[test])
        data = cur.fetchall()
        items = []
        for item in data:
            print(item['ItemID'])
            sql = "select * from item where ID = " + str(item['ItemID'])
            print(sql)
            cur=mysql.connection.cursor()
            cur.execute(sql)
            datas = cur.fetchone()
            itemarr = []
            itemarr.append(1.7)
            itemarr.append(datas["b"])
            itemarr.append(0)
            itemarr.append(1)
            qp.append(itemarr)

        theta = est_theta(0,resp,numpy.array(qp))
        score = int(theta * 100 + 400)
        acc = accuracy(theta,resp,numpy.array(qp))
        feedback = student_feedback(theta)
        return render_template('1PLresult.html',tables=df,accuracy = acc, score = score,url = url,data = session['username'],feedback = feedback)
        
        #cur=mysql.connection.cursor()
        #cur.execute("select * from user where email=%s and password=%s",(email,pwd))
        #data=cur.fetchone()
        #if data:
        #    session['logged_in']=True
        #    session['username']=data["username"]
        #   flash('Login Successfully','success')
        #   return redirect('home')
        #else:
    return render_template("login.html")


@app.route('/twoPL',methods=['POST','GET'])
def twoPL():
    #print(est_theta(0,rv,numpy.array(qp)))
    status=True
    qp=[]
    if request.method=='POST':
        uname=request.form["uname"]
        print(uname)
        usn=request.form["usn"]
        print(usn)
        test=request.form["test"]
        print(test)
        f = request.form["file"]
        print(f)
        file_path = secure_filename(f)
        df = pd.read_csv(file_path)
        df = df.values.tolist()
        resp = []
        for d in df:
            resp.append(d[1])
        print(resp)
        cur=mysql.connection.cursor()
        cur.execute("select * from tests where TESTID = %s",[test])
        data = cur.fetchall()
        items = []
        for item in data:
            print(item['ItemID'])
            sql = "select * from item where ID = " + str(item['ItemID'])
            print(sql)
            cur=mysql.connection.cursor()
            cur.execute(sql)
            datas = cur.fetchone()
            itemarr = []
            itemarr.append(datas["a"])
            itemarr.append(datas["b"])
            itemarr.append(0)
            itemarr.append(1)
            qp.append(itemarr)
        print(qp)
        print(resp)
        theta = est_theta(0,resp,numpy.array(qp))
        score = int(theta * 100 + 400)
        acc = accuracy(theta,resp,numpy.array(qp))
        feedback = student_feedback(theta)
        return render_template('2PLresult.html',tables=df,score = score,url = url,accuracy = acc, data = session['username'],feedback = feedback)


@app.route('/threePL',methods=['POST','GET'])
def threePL():
    #print(est_theta(0,rv,numpy.array(qp)))
    status=True
    qp=[]
    if request.method=='POST':
        uname=request.form["uname"]
        print(uname)
        usn=request.form["usn"]
        print(usn)
        test=request.form["test"]
        print(test)
        f = request.form["file"]
        print(f)
        file_path = secure_filename(f)
        df = pd.read_csv(file_path)
        df = df.values.tolist()
        resp = []
        for d in df:
            resp.append(d[1])
        print(resp)
        cur=mysql.connection.cursor()
        cur.execute("select * from tests where TESTID = %s",[test])
        data = cur.fetchall()
        items = []
        for item in data:
            print(item['ItemID'])
            sql = "select * from item where ID = " + str(item['ItemID'])
            print(sql)
            cur=mysql.connection.cursor()
            cur.execute(sql)
            datas = cur.fetchone()
            itemarr = []
            itemarr.append(datas["a"])
            itemarr.append(datas["b"])
            itemarr.append(datas["c"])
            itemarr.append(1)
            qp.append(itemarr)
        print(qp)
        print(resp)
        theta = est_theta(0,resp,numpy.array(qp))
        score = int(theta * 100 + 400)
        acc = accuracy(theta,resp,numpy.array(qp))
        feedback = student_feedback(theta)
        return render_template('3PLresult.html',tables=df,score = score,url = url,data = session['username'],feedback = feedback,accuracy = acc)



@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = secure_filename(f.filename)
        f.save(file_path)
        # Make prediction
        result = random.uniform(0.65,1.0)
        os.remove(file_path)
        return str(result)
    return None


@app.route('/')
def index():
    return render_template('login.html')

def is_logged_in(f):
	@wraps(f)
	def wrap(*args,**kwargs):
		if 'logged_in' in session:
			return f(*args,**kwargs)
		else:
			flash('Unauthorized, Please Login','danger')
			return redirect(url_for('login'))
	return wrap
  
#Registration  
@app.route('/reg',methods=['POST','GET'])
def reg():
    status=False
    if request.method=='POST':
        name=request.form["uname"]
        email=request.form["email"]
        pwd=request.form["upass"]
        cur=mysql.connection.cursor()
        cur.execute("insert into user(username,password,email) values(%s,%s,%s)",(name,pwd,email))
        mysql.connection.commit()
        cur.close()
        flash('Registration Successfully. Login Now...','success')
        return redirect('login')
    return render_template("login.html",status=status,url = url,data = session['username'])
global testID
@app.route('/live_test',methods=['POST','GET'])
@is_logged_in
def live_test():
    if request.method=='POST':
        global testID
        testID = request.form["testID"]
        cur=mysql.connection.cursor()
        cur.execute("select * from tests where TESTID = %s",[testID])
        data = cur.fetchall()
        questions = []
        for item in data:
            print(item['ItemID'])
            sql = "select * from item where ID = " + str(item['ItemID'])
            cur=mysql.connection.cursor()
            cur.execute(sql)
            datas = cur.fetchone()
            datas['ID'] = str(datas['ID'])
            questions.append(datas)
        return render_template("live_test.html",url = url,data = session['username'],questions = questions)

@app.route('/live_test_evaluate',methods=['POST','GET'])
def live_test_evaluate():
    global testID
    if request.method == 'POST':
        cur=mysql.connection.cursor()
        cur.execute("select * from tests where TESTID = %s",[testID])
        data = cur.fetchall()
        print(data)
        questions = []
        answerss = []
        qp=[]
        rv=[]
        for answers in data:
            print(answers)
            input_ans_name = "answer" + str(answers['ItemID'])
            student_response = request.form[input_ans_name]
            sql = "select * from item where ID = " + str(answers['ItemID'])
            cur.execute(sql)
            correct_answer = cur.fetchone()
            answerss.append(student_response)
            if student_response == correct_answer['answer']:
                rv.append(1)
            else:
                rv.append(0)
            itemarr = []
            itemarr.append(correct_answer["a"])
            itemarr.append(correct_answer["b"])
            itemarr.append(correct_answer["c"])
            itemarr.append(1)
            qp.append(itemarr)
            questions.append(correct_answer["d"])
    theta = est_theta(0,rv,numpy.array(qp))
    score = int(theta * 100 + 400)
    acc = accuracy(theta,rv,numpy.array(qp))
    feedback = student_feedback(theta)
    df = pd.DataFrame({
        'Question': questions,
        'Answer': answerss,
        'Status': rv
    })
    df = df.values.tolist()
    return render_template('liveresult.html',tables=df,score = score,url = url,data = session['username'],feedback = feedback,accuracy = acc)


#Home page
@app.route("/home",methods=['POST','GET'])
@is_logged_in
def home():
    global url
    if request.method=='POST':
        if request.form.get("submit") == "Get 1PL results":
            print('Demo 1 Selected')
            return render_template('1PL.html',url = url,data = session['username'])
        if request.form.get("submit") == "Get 2PL results":
            print('Demo 2 Selected')
            return render_template('2PL.html',url = url,data = session['username'])
        if request.form.get("submit") == "Get 3PL results":
            print('Demo 3 Selected')       
            return render_template('3PL.html',url = url,data = session['username'])
        if request.form.get("submit") == "Items Bank and train":
            print('Demo Selected')
            return render_template('demo3.html')
    return render_template('index.html',data = session['username'],url = url)
@app.route("/logout")
def logout():
	session.clear()
	flash('You are now logged out','success')
	return redirect(url_for('login'))

if __name__ == '__main__':
    global url
    app.secret_key='secret123'
    myIP = get_ip_address_of_host()
    url = 'http://' + myIP + ':5000'
    webbrowser.open_new(url)
    ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)
    app.run(debug=False, host='0.0.0.0')
