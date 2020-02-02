import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import metrics
import collections
import sys


def categoryTransform(d):
    CategoryString = d["Category"]
    categoryVal = d["Category"].unique()
    categoryValCount = len(categoryVal)
    category_dict = {}
    for i in range(0, categoryValCount):
        category_dict[categoryVal[i]] = i
    d["Category_i"] = d["Category"].map(category_dict).astype(int)

def newInstallsTrans(d):
    CategoryString = d["Installs"]
    categoryVal = d["Installs"].unique()
    categoryVal = sorted(categoryVal)
    categoryValCount = len(categoryVal)
    category_dict = dict()
    for i in range(0, categoryValCount):
        category_dict[categoryVal[i]] = i
    #d["Installs"] = d["Installs"].map(category_dict).astype(int)
    
def size_trans(size):
    if 'M' in size:
        x = size[:-1]
        x = float(x)*1000000
        return(x)
    elif 'k' == size[-1:]:
        x = size[:-1]
        x = float(x)*1000
        return(x)
    else:
        return None
    
def sizeTransform(d):
    d["Size"] = d["Size"].map(size_trans)
    data.Size.fillna(method = 'ffill', inplace = True)
    
def installs_trans(inst):
    try:
        x = (inst[:-1].replace(",",""))
        x = int(x)
        #return x
    except ValueError:
        print("Line is corrupt!", x)
        x = 0
    return x
def installsTransform(d):
    d["Installs"] = d["Installs"].map(installs_trans)

def type_trans(types):
    if types == 'Free':
        return 0
    else:
        return 1
def typeTransfor(d):
    d["Type"] = d["Type"].map(type_trans)

def ratingTransform(d):
    RatingL = d["Content Rating"].unique()
    RatingDict = {}
    for i in range(len(RatingL)):
        RatingDict[RatingL[i]] = i
    
    d["Content Rating"] = d["Content Rating"].map(RatingDict).astype(int)

def genreTranform(d):
    GenresL = d.Genres.unique()
    GenresDict = {}
    for i in range(len(GenresL)):
        GenresDict[GenresL[i]] = i

    d["Genres_i"] = d["Genres"].map(GenresDict).astype(int)
    
def price_trans(price):
    if price == '0':
        return 0
    else:
        price = price[1:]
        if price == "veryone":
            price = 0
        price = float(price)
        return price
def priceTransform(d):
    d["Price"] = d["Price"].map(price_trans).astype(float)
    
def review_trans(rev):
    if 'M' in rev:
        x = rev[:-1]
        x = float(x)*1000000
        return(x)
    elif 'k' == rev[-1:]:
        x = rev[:-1]
        x = float(x)*1000
        return(x)
    else:
        return rev

def reviewTrans(d):
    d["Reviews"] = d["Reviews"].map(review_trans).astype(float)

def name_size(name):
    return len(name.split())

def nameSize(d):
    d["NameSize"] = d["App"].map(name_size).astype(int)
    
def transformData(d):
    categoryTransform(d)
  
    sizeTransform(d)
    d.Size.fillna(method = 'ffill', inplace = True)
    installsTransform(d)
    newInstallsTrans(d)
    typeTransfor(d)
    ratingTransform(d)
    genreTranform(d)
    priceTransform(d)
    reviewTrans(d)
    nameSize(d)
    d.drop(labels = ["Genres","Category","Last Updated","Current Ver", 'Android Ver', 'App'], axis = 1, inplace = True)

data = pd.read_csv('./data/googleplaystore.csv')
data = data.dropna()
data = data.reset_index(drop=True)
transformData(data)

data.info()

alpha = 0.005

X = data.drop(labels = ['Price','Rating','Category_i','Type','Content Rating','Size','Genres_i','NameSize'],axis = 1)
Y = data.Rating


m = np.shape(X)[0]
n = np.shape(X)[1]
X = np.hstack([np.ones([m, 1]), X])


def h(x, Theta):
    return np.dot(x, np.transpose(Theta))

def coste(X, Y, Theta):
    H = h(X, Theta)
    Aux = (H - Y) ** 2
    return (Aux.sum() / (2 * len(X)))

def descenso_gradiente(x, y, times):
    theta = np.zeros([1, n+1], dtype = float)
    c = np.zeros([times, 1], dtype = float)
    for i in range(times):
        H = h(x, theta)
        aux0 = theta[0][0] - alpha*np.sum((H-y))*(1/m)
        aux1 = theta[0][1] - alpha*np.sum((H-y)*x)*(1/m)
        theta[0][0] = aux0
        theta[0][1] = aux1
        c[i] = coste(x, y, theta)
    return [theta, c]
#X = np.transpose(X)



Theta = descenso_gradiente(X, Y,500)

plt.plot(X[:, 1:], h(X, Theta[0]))