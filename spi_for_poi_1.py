import math
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors.kde import KernelDensity

# to estimate probability density for each feature
def density_estimate(data):
    x = [[i] for i in data]
    bandwidth = 10**np.linspace(-1,1,100)
    grid = GridSearchCV(KernelDensity(kernel = 'gaussian'),{'bandwidth':bandwidth},cv = 10).fit(x)
    h = grid.best_index_
    kde = KernelDensity(kernel = 'gaussian',bandwidth= bandwidth[h]).fit(x)
    return kde

# to train the model
def train_model(df):
    data = [[],[],[],[],[],[],[],[],[],[]]
    # fetching record one by one
    for row in df.iterrows():
        index, record = row
        if(index>=100):
            break
        else:
            # complete record for each feature
            for i in range(10):
                data[i].append(record[i+1])
    # train the model for each feature
    probability = []
    for i in range(10):
        probability.append(density_estimate(data[i]))
    # pdf for each feature computed
    return probability

#to calculate rank
def btmodel(probability):
    score = [0]
    for i in range(len(probability)-1):
        score.append(math.log(probability[i+1]/probability[i])+score[-1])
    return score

def point_of_interest(df, func):
    probability = []
    for row in df.iterrows():
        index, record = row
        p = 1
        for i in range(10):
            p = p*math.exp(func[i].score_samples(record[i+1]))
        probability.append(p)
    return btmodel(probability)
    #return probability


#reading csv file
def extractfromcsv(fname):
    df = pd.read_csv(fname)
    return df

def get_ranked_data(fname):
    df = extractfromcsv(fname) # to read from csv file
    func = train_model(df)
    score = point_of_interest(df,func)
    df['bt_score'] = pd.Series(score)

    # to compute mean squared error in ranking
    rank = [x for _, x in sorted(zip(score, list(df['rank'])), reverse=True)]
    error = 0
    for i in range(100,len(rank)):
        error = error + abs(i+1-rank[i])
    print(error/(len(rank)-100)) # printing error in ranking
    return df

fname = input("enter file name")
print(get_ranked_data(fname))
#to final output

