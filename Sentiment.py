import numpy as np
import re
import time
start = time.time()


dct = {} # dictionary
train_rec_num = 700
SIZE = 5000

articles_itp = {'a', 'an', 'the', 'is', 'was', 'are', 'were', 'will', 'be', 'am', 're', 'and', 'or', 'to', 's', 'it'}

for num in range(0, train_rec_num):
    str_num = str(num).zfill(3)  #001
    path = f"train_rec_pos/train_p{str_num}"
    with open(path, 'r', encoding='utf-8') as f: #reading file
        text = f.read()
    words = re.findall(r'[A-Za-z0-9]+', text.lower()) #only words and numbers
    s = set(words)
    for word in s: #adding to dict
        if((word not in dct) and (word not in articles_itp)):
            dct[word] = 1
        elif (word not in articles_itp):
            dct[word] += 1

for num in range(0, train_rec_num):
    str_num = str(num).zfill(3)  #001
    path = f"train_rec_neg/train_n{str_num}"
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    words = re.findall(r'[A-Za-z0-9]+', text.lower())
    s = set(words)
    for word in s:
        if((word not in dct) and (word not in articles_itp)):
            dct[word] = 1
        elif (word not in articles_itp):
            dct[word] += 1

dct = dict(sorted(dct.items(), key=lambda x: x[1], reverse=True)[:SIZE]) #sortowanie po najczestszemu spotykaniu, odebranie 5000
size = len(dct)

idx = 0
for p in dct: #indeksing the dictionary
    dct[p] = idx
    idx += 1

vector_p = np.zeros((train_rec_num, SIZE), dtype=float) # matryca pos values

for num in range(0, train_rec_num): # stw wekt [0,1,0,0,...,1,0]
    str_num = str(num).zfill(3)  #001
    path = f"train_rec_pos/train_p{str_num}"
    with open(path, 'r', encoding='utf-8') as f: #reading file
        text = f.read()
    words = re.findall(r'[A-Za-z0-9]+', text.lower()) #only words and numbers
    s = set(words)
    for word in s:
        if(word in dct):
            vector_p[num, dct[word]] += 1
    norm = np.linalg.norm(vector_p[num]) # L2 norma dla latwiejszego przechowywania danych
    if(norm != 0):
        vector_p[num] /= norm

vector_n = np.zeros((train_rec_num, SIZE), dtype=float) # matryca neg values

for num in range(0, train_rec_num): # stw wekt [0,1,0,0,...,1,0]
    str_num = str(num).zfill(3)  #001
    path = f"train_rec_neg/train_n{str_num}"
    with open(path, 'r', encoding='utf-8') as f: #reading file
        text = f.read()
    words = re.findall(r'[A-Za-z0-9]+', text.lower()) #only words and numbers
    s = set(words)
    for word in s:
        if(word in dct):
            vector_n[num, dct[word]] += 1
    norm = np.linalg.norm(vector_n[num])
    if (norm != 0):
        vector_n[num] /= norm

all_data = np.vstack((vector_p, vector_n)) # uniting the whole data in one matrix (vstack -- vertical)
p_rec = np.ones(train_rec_num)
n_rec = np.zeros(train_rec_num)
all_rec = np.hstack((p_rec, n_rec)) # uniting their values (hstack -- horizontal)

def a(preactivation): # funkcja aktywacji sygmoida
    activation = 1/(1+np.exp(-preactivation))
    return activation

def comp_loss(y, y_pred):
   return np.sum((y - y_pred)**2)

np.random.seed(42)
w = np.random.randn(SIZE) * 1e-3 # wektor w: z = w*x + b, ocenia waznosc kazdego slowa
b = 0.0 # bias

def training(w, b, l_rate = 5, X = all_data, y = all_rec, epochs = 10):
    for epoch in range(epochs):
        i = np.random.permutation(train_rec_num*2) # randomowy porzadek trainingow
        X = X[i]
        y = y[i]
        y_pred = np.zeros(train_rec_num*2)
        for num in range(0, train_rec_num*2):
            z = np.dot(X[num], w) + b
            y_i = a(z)
            y_pred[num] = y_i
            er = y_i - y[num] # error
            w -= l_rate * er * X[num]
            b -= l_rate * er
        print("Loss on epoch", epoch + 1, ":", comp_loss(y, y_pred))
    return w, b

w, b = training(w, b)

test_rec_num = 600

vector_t = np.zeros((test_rec_num, SIZE), dtype=float) # matryca pos values

for num in range(0, test_rec_num): # stw wekt [0,1,0,0,...,1,0]
    str_num = str(num)  #0
    path = f"test_rec/rec_{str_num}"
    with open(path, 'r', encoding='utf-8') as f: #reading file
        text = f.read()
    words = re.findall(r'[A-Za-z0-9]+', text.lower()) #only words and numbers
    s = set(words)
    for word in s:
        if(word in dct):
            vector_t[num, dct[word]] += 1
    norm = np.linalg.norm(vector_t[num]) # L2 norma dla latwiejszego przechowywania danych
    if(norm != 0):
        vector_t[num] /= norm

def predictor(w, b, X = vector_t): #tworzenie matrycy dla test_rec
    y_pred = np.zeros(test_rec_num)
    for num in range(0, test_rec_num):
        z = np.dot(X[num], w) + b
        y_i = a(z)
        if(y_i >= 0.5):
            y_i = 1
        else:
            y_i = -1
        y_pred[num] = y_i
    return y_pred

y_pred = predictor(w, b)

def checker(y_pred): #sprawdzanie prawidlowosci predictora
    path = f"oceny_test_rec.out"
    er_num = 0
    with open(path, 'r', encoding='utf-8') as f: #reading file
        text = f.read()
        text = text.replace('rec_', '')
    for line in text.splitlines():
        check_num, val = line.split()
        check_num = int(check_num)
        val = int(val)
        if(y_pred[check_num] != val):
            er_num += 1
    return er_num

er_num = checker(y_pred)
accuracy = 100*(test_rec_num-er_num) / test_rec_num
end = time.time()

print(f"Model accuracy: {accuracy:.2f}%")
print(f"Time: {end-start:.2f} seconds")