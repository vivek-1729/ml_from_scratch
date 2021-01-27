import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
df = pd.read_excel('data.xlsx', sheet_name='X', header=None)

def probability(df):
    mu = np.sum(df, axis=0) / len(df)
    variance = np.sum((df - mu)**2, axis=0) / len(df)
    var_dia = np.diag(variance) #adds the variances to the diagonal of the matrix, is this the correlation matrix?
    k = len(mu)
    X = df - mu
    p = 1/((2*np.pi)**(k/2)*(np.linalg.det(var_dia)**0.5))* np.exp(-0.5* np.sum(X @ np.linalg.pinv(var_dia) * X,axis=1)) #multiply (X- Xbar) by the gaussian probability function
    return p

def tpfpfn(ep, p, y):
    tp, fp, fn = 0, 0, 0
    for i in range(len(y)):
        if p[i] <= ep and y[i][0] == 1:
            tp += 1
        elif p[i] <= ep and y[i][0] == 0:
            fp += 1
        elif p[i] > ep and y[i][0] == 1:
            fn += 1
    return tp, fp, fn

def f1(ep, p):
    tp, fp, fn = tpfpfn(ep, p1, cvy)
    prec = tp/(tp + fp)
    rec = tp/(tp + fn)
    f1 = 2*prec*rec/(prec + rec)
    return f1

cvx = pd.read_excel('data.xlsx', sheet_name='Xval', header=None)
cvy = np.array(pd.read_excel('data.xlsx', sheet_name='y', header=None))
p1 = probability(cvx)
eps = [i for i in p1 if i <= p1.mean()]
f = []
for i in eps:
    f.append(f1(i, p1))
e = eps[np.array(f).argmax()] #epilson is chosen as the probability of the point with the biggest f-score in the eps array (all points with probability below mean)
normal = pd.DataFrame(columns=(0, 1))
anomolous = pd.DataFrame(columns=(0, 1))
for i in range(len(df)):
    if p1[i] <= e:
        anomolous.loc[i] = df.iloc[i]
    else:
        normal.loc[i] = df.iloc[i]
plt.scatter(anomolous[0], anomolous[1], color='r')
plt.scatter(normal[0], normal[1], color='k')
plt.show()