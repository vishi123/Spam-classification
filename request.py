# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 11:54:29 2020

@author: Manvi Gupta
"""

import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'v2':"This is for your new jio number"})

print(r.json())