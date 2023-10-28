﻿### [ Practice Module ] Project Submission Template: Github Repository & Zip File

**[ Naming Convention ]** CourseCode-StartDate-BatchCode-TeamName-ProjectName.zip

* **[ MTech Thru-Train Group Project Naming Example ]** IRS-PM-2020-01-18-IS02PT-GRP-AwsomeSG-HDB_BTO_Recommender.zip

* **[ MTech Stackable Group Project Naming Example ]** IRS-PM-2020-01-18-STK02-GRP-AwsomeSG-HDB_BTO_Recommender.zip

[Online editor for this README.md markdown file](https://pandao.github.io/editor.md/en.html "pandao")

---

### <<<<<<<<<<<<<<<<<<<< Start of Template >>>>>>>>>>>>>>>>>>>>

---

## SECTION 1 : PROJECT TITLE
## PostCraft: X Tweet Features Recommendation and Text Generation System

<img src="ProjectCode/static/images/introduction page1.png"
     style="float: left; margin-right: 0px;" />

---

## SECTION 2 : ABSTRACT
With the rise of social media platforms, the landscape of digital marketing has undergone significant transformations. More and more brands and individuals are utilizing social media for both  product or service promotion. Platforms like X have become crucial spaces for businesses to showcase their products and engage with potential customers. However, the overcrowded social media space makes it challenging for brands to stand out and effectively reach their target audience. While various data analytics tools are available in the current market, they still require experienced professionals for analysis and creativity in marketing, thereby increasing human resource and time costs. Moreover, existing solutions might not fully meet the unique and real-time promotional needs faced by different good categories in the highly competitive and ever-changing environment.

To address this market pain, we have developed PostCraft, a data-driven tweet feature and text recommendation and generation system. The system aims to provide efficient and intelligent marketing solutions for both individuals and businesses. Postcraft comprises three main components: Tweets Feature Recommender, Image Content Recommender, and Tweet Optimizer. This system integrates multiple machine learning algorithms, NLP techniques, various recommendation algorithms, and LLM tools to establish a comprehensive platform. By analyzing current market trends and key components of popular posts, Postcraft facilitates tweet feature recommendation, tweet content optimization and generation, as well as image content recommendations for tweets.

In terms of performance validation, we sent out 180 tweets from 100 accounts, with 90 generated by PostCraft and 90 by ChatGPT. We use viewing count within a 3-day period as metrics. Using a t-test and assuming equal means for both methods' outcomes, at a significance level of 0.05,we obtained a p-value of 0.0346 and a t-statistic of 2.1293. Therefore, we reject the hypothesis and conclude that tweets generated by PostCraft significantly outperformed those generated by ChatGPT . This finding strongly supports the outstanding performance of our system in social media tweet generation.

PostCraft has been deployed on a website, enabling users to conveniently access it online. It utilizes Flask as the application framework, HTML/CSS/JavaScript for frontend development to create an intuitive user interface, and Python as the backend to generate recommendations requested through Flask.

---

## SECTION 3 : CREDITS / PROJECT CONTRIBUTION

| Official Full Name  | Student ID (MTech Applicable)  | Work Items (Who Did What) | Email (Optional) |
| :------------ |:---------------:| :-----| :-----|
| Liu Siyan | A0285857H | Model Design and construction, Back-end developent and Integration, Data collation, cleansing and preliminary analysis, Practical validation, Documentation and Figures| E1221669@u.nus.edu|
| Lin Zijun | A0285897Y | Front end development, Front end and back end Integration, Deployment, User Guide| A1234567B@gmail.com |
| Lai Weichih | A0285875H | Data collection and crawling, Back-end development, Documentation| E1221687@u.nus.edu|
| Fang Ruolin | A0285983H| Business analysis, Data collection and crawling, Documentation| E1221795@u.nus.edu|

---

## SECTION 4 : VIDEO OF SYSTEM MODELLING & USE CASE DEMO

Market Research; Demo
https://youtu.be/1lRf8VBaKcI

System Design; Technical explanation of use cases
https://youtu.be/of07CCpwa1s 
---

## SECTION 5 : USER GUIDE

`Refer to appendix <Installation & User Guide> in project report at Github Folder: ProjectReport`

### [ 1 ] To run the system using iss-vm

> download pre-built virtual machine from http://bit.ly/iss-vm

> start iss-vm

> open terminal in iss-vm

> $ git clone https://github.com/telescopeuser/Workshop-Project-Submission-Template.git

> $ source activate iss-env-py2

> (iss-env-py2) $ cd Workshop-Project-Submission-Template/SystemCode/clips

> (iss-env-py2) $ python app.py

> **Go to URL using web browser** http://0.0.0.0:5000 or http://127.0.0.1:5000

### [ 2 ] To run the system in other/local machine:
### Install additional necessary libraries. This application works in python 3 only.

> $ sudo apt-get install python-clips clips build-essential libssl-dev libffi-dev python-dev python-pip

> $ pip install pyclips flask flask-socketio eventlet simplejson pandas

---
## SECTION 6 : PROJECT REPORT / PAPER

`Refer to project report at Github Folder: ProjectReport`

---
## SECTION 7 : MISCELLANEOUS

`Refer to Github Folder: Miscellaneous`

---

### <<<<<<<<<<<<<<<<<<<< End of Template >>>>>>>>>>>>>>>>>>>>

---

**This [Machine Reasoning (MR)](https://www.iss.nus.edu.sg/executive-education/course/detail/machine-reasoning "Machine Reasoning") course is part of the Analytics and Intelligent Systems and Graduate Certificate in [Intelligent Reasoning Systems (IRS)](https://www.iss.nus.edu.sg/stackable-certificate-programmes/intelligent-systems "Intelligent Reasoning Systems") series offered by [NUS-ISS](https://www.iss.nus.edu.sg "Institute of Systems Science, National University of Singapore").**

**Lecturer: [GU Zhan (Sam)](https://www.iss.nus.edu.sg/about-us/staff/detail/201/GU%20Zhan "GU Zhan (Sam)")**

[![alt text](https://www.iss.nus.edu.sg/images/default-source/About-Us/7.6.1-teaching-staff/sam-website.tmb-.png "Let's check Sam' profile page")](https://www.iss.nus.edu.sg/about-us/staff/detail/201/GU%20Zhan)

**zhan.gu@nus.edu.sg**
