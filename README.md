# Item-Response-Theory-based-student-evaluation
Flask-based web application for Item Response Theory (IRT) analysis that supports 1PL, 2PL, and 3PL test scoring using MySQL-backed item banks, CSV response uploads, and user authentication.

IRT Test Scoring Web Application

Overview
This project is a Flask-based web application designed to perform Item Response Theory (IRT) analysis for educational and psychometric testing. The system supports 1PL, 2PL, and 3PL models to estimate candidate ability and generate test scores using uploaded response data and database-stored item banks.

Features

User authentication and session management

Support for 1PL (Rasch Model)

Support for 2PL (Two Parameter Logistic Model)

Support for 3PL (Three Parameter Logistic Model)

CSV-based response file upload

MySQL-backed item bank and test configuration

Ability estimation using Maximum Likelihood Estimation

Automated test score calculation

Secure file upload and handling

Web dashboard for displaying results

Supported IRT Models
1PL Model
Uses difficulty parameter (b)

2PL Model
Uses discrimination (a) and difficulty (b)

3PL Model
Uses discrimination (a), difficulty (b), and guessing parameter (c)

Technology Stack

Backend

Python Flask

NumPy

Pandas

NumExpr

Flask-MySQLdb

Frontend

HTML

CSS

JavaScript

Jinja2 Templates

Database

MySQL database storing users, item bank, and test configuration

Project Workflow

User logs into the system

User selects IRT model (1PL, 2PL, or 3PL)

User uploads response CSV file

Application retrieves item parameters from database

Ability (theta) is estimated using likelihood functions

Final score is calculated and displayed

Input File Format
Response file must be in CSV format containing:

Question ID

Candidate response (binary or boolean)

Installation Steps

Clone Repository
git clone <repository-url>
cd <repository-folder>

Install Dependencies
pip install flask flask-mysqldb numpy pandas numexpr pillow tqdm

Configure Database
Update database credentials inside the application configuration:
MYSQL_HOST
MYSQL_USER
MYSQL_PASSWORD
MYSQL_DB

Run Application
python app.py

Application Access
The application runs locally and automatically opens in a web browser.

Authentication

Supports user registration

Session-based login system

Protected dashboard routes

Scoring Methodology

Uses logistic Item Characteristic Curve functions

Maximum Likelihood Estimation for ability calculation

Score transformation applied for final result generation

Future Enhancements

Computer Adaptive Testing support

Advanced analytics dashboard

Machine learning based ability prediction

REST API integration

Role-based access control

Contribution
Contributions, bug fixes, and feature suggestions are welcome.

License
This project is intended for educational and research purposes.