import csv
import numpy as np
import sys

feature=106
def sigmoid(z):
    res = 1/(1.0+np.exp(-z))
    return np.clip(res, 0.00000000000001, 0.99999999999999)
w = [-3.87844595e-01, 1.85352923e+00,  9.76109494e-01, 8.64609516e-01,
     2.67560997e+01, 2.66998404e+00, 2.83563710e+00, 1.24569088e-01,
     -5.80168534e-01,  -2.03067386e+00, -3.60645240e-01, -1.78994557e-01,
     -8.32629497e-01, -6.66415036e-01, -5.55620367e+00, -8.73483002e-01,
     -1.60930178e+00, -1.55981943e+00, -1.06301099e+00, -2.20208716e+00,
     -2.09943895e+00, -2.12922510e+00 ,- 1.90852390e+00, -2.31552283e-01,
     -2.16284413e-01, 4.26105775e-01,  1.54728995e+00, -7.99711157e-01,
     8.02595004e-01, -9.71022131e+00,  1.29264089e+00, -4.55781890e-01,
     -1.53419002e+00, 7.36496008e-01,  2.69550500e-01, -1.58839211e+00,
     -1.97926918e+00, -1.60100309e+00, -1.47127131e+00, -6.78687591e-01,
     -1.82356391e+00, -6.09311241e-01,   6.21665515e-02, -1.70889927e+00,
     -1.37026841e+00, -1.00661447e+00, -1.55729866e+00, -4.01069262e+00,
     -1.64789643e-01, -1.42608219e-01, -4.15809935e-01,  2.53499830e-03,
     -7.88389592e-01, -8.81118414e-01, -5.17541329e-01, -3.36070883e-01,
     -1.25658084e+00, -1.52770159e+00, -4.73096847e-01,  8.63093394e-01,
     -1.00109276e+00, -4.56991013e-01, -6.80860982e-01, -9.69372407e-01,
     -5.13364583e-01, 4.45450075e-01, -3.00421391e-01, -1.37421276e+00,
     -2.64084132e+00, -3.65086613e-01, -2.51605233e+00, -2.11149742e+00,
     -1.09482918e+00, -3.48073019e-01, -1.21185140e-02, -1.62025452e-01,
     -1.59667973e+00, -2.68930437e+00, -7.53049750e-01, -2.04620533e+00,
     -1.91500718e+00, -3.71132677e-01, -5.86251526e-01, -1.12621166e+00,
     -5.46873193e-01, -3.13860890e-02,  2.56424204e-01, -5.14148011e-01,
     -3.97501557e-01, -8.42020259e-01, -1.12493190e+00, -1.50197435e+00,
     -5.48822719e+00, -1.51122180e+00, -2.64259059e-01, -4.74814585e-01,
     -6.26252196e-01, -9.95761916e-01, -8.78919045e-01, -1.74350223e+00,
     -6.09548225e-01, -1.21016074e+00, -7.49625058e-01, -4.90841224e-01,
    -1.73572109e+00, -5.29314819e-02, -9.35838386e-01]

w=np.array(w)
b = -0.541029228846
test = []
rowtest = csv.reader(open(sys.argv[1], 'r'), delimiter=",")
rowtest_special = csv.reader(open(sys.argv[2], 'r'), delimiter=",")
# get special feature from test.csv
test_special = []
i = -1
for r in rowtest_special:
    if i == -1:
        i = 0
        continue
    test_special.append(r[4])

i = -1  # no data ay first row
for r in rowtest:
    if i == -1:
        i = 0
        continue
    test.append([])
    test[i].append(float(test_special[i]))
    for j in range(feature):
        test[i].append(float(r[j]))
    i += 1

test = np.array(test).transpose()

test_max = np.amax(test, axis=1)
test_min = np.amin(test, axis=1)
for i in range(7):
    if i != 3:
        test[i] = (test[i]-np.outer(test_min[i], np.ones(test.shape[1])))/(np.outer(test_max[i]-test_min[i], np.ones(test.shape[1])))


z = np.dot(w.reshape(1, feature+1), test)+b
est = sigmoid(-z)
est = est.reshape(est.shape[1])
pre = []
for i in range(len(est)):  # start from 1
    pre.append([str(i+1)])
    if est[i] > 0.5:
        pre[i].append(0)
    else:
        pre[i].append(1)
filename = sys.argv[3]
text = open(filename, "w+")
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["id", "label"])
for i in range(len(est)):
    s.writerow(pre[i])
text.close()
print("Success")
