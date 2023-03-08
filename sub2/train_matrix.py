import pandas as pd
import time
import onnxmltools
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.gaussian_process import  GaussianProcessRegressor,GaussianProcessClassifier
from sklearn import svm
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType



sum_year={}
sum_num={}
sum_month={}
FILE= open('./CODA_TB_Solicited_Meta_Info.csv', 'r')
FILE.readline()
for line  in FILE:
    line=line.strip()
    table=line.split(',')
    pid=table[0]
    ttt=table[1].split('-recording')
    my_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(float(ttt[0])/1000)))
    t=my_time.split(' ')
    t=t[0].split('-')
    if pid in sum_year:
        sum_year[pid]=sum_year[pid]+float(t[0])
        sum_num[pid]=sum_num[pid]+1
        sum_month[pid]=sum_month[pid]+float(t[1])
    else:
        sum_year[pid]=float(t[0])
        sum_num[pid]=1
        sum_month[pid]=float(t[1])
FILE.close()

FILE = open('/input/meta_info.csv', 'r')
#FILE = open('/input/meta_info.csv', 'r')
FILE.readline()
for line  in FILE:
    line=line.strip()
    table=line.split(',')
    pid=table[0]
    ttt=table[1].split('-recording')
    my_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(float(ttt[0])/1000)))
    t=my_time.split(' ')
    t=t[0].split('-')
    if pid in sum_year:
        sum_year[pid]=sum_year[pid]+float(t[0])
        sum_num[pid]=sum_num[pid]+1
        sum_month[pid]=sum_month[pid]+float(t[1])
    else:
        sum_year[pid]=float(t[0])
        sum_num[pid]=1
        sum_month[pid]=float(t[1])



df=pd.read_csv('./CODA_TB_Solicited_Meta_Info.csv')
count=df['participant'].value_counts()
sound_max = df[['participant', 'sound_prediction_score']].groupby('participant').max()

FILE = open('./CODA_TB_Clinical_Meta_Info_Train.csv', 'r')
l=FILE.readline()
matrix=[]
gs=[]
for l in FILE:
    l = l.strip()
    values= l.split(',')
    
    name=values.pop(0)
    label=values.pop(-1)
    gs.append(label)
    vector=[]
    for val in values:
        if val == 'Male':
            val=1
        elif val == 'Female':
            val=0
        elif val == 'Yes':
            val=1
        elif val == 'No':
            val=0
        elif val == 'Not sure':
            val=np.nan
        else:
            try:
                val=float(val)
            except:
                val=np.nan
                print(val)
        vector.append(val)

    try:
        vector.append(count[name])
    except:
        vector.append(0)
    try:
        vector.append(sound_max.loc[name]['sound_prediction_score'])
    except:
        vector.append(0)
    try:
        #print(name)
        avg=sum_month[name]/sum_num[name]
        vector.append(avg)
    except:
        print(name)
        vector.append(6)
        print('not exist')
    try:
        avg=sum_year[name]/sum_num[name]
        vector.append(avg)
    except:
        print(name)
        vector.append(2022)
        print('not exist')

    vector=np.asarray(vector)
    matrix.append(vector)

matrix=np.asarray(matrix)
matrix = (matrix - np.mean(matrix, axis = 0))/(np.std(matrix, axis =0)+1e-6)
matrix= np.float32(matrix)


matrix_nonan = []
FILE = open('./CODA_TB_Clinical_Meta_Info_Train.csv', 'r')
l=FILE.readline()
for l in FILE:
    l = l.strip()
    values= l.split(',')
    
    name=values.pop(0)
    label=values.pop(-1)
    vector=[]
    for val in values:
        if val == 'Male':
            val=1
        elif val == 'Female':
            val=0
        elif val == 'Yes':
            val=1
        elif val == 'No':
            val=0
        elif val == 'Not sure':
            val=0.5
        else:
            try:
                val=float(val)
            except:
                val= 0
                print(val)
        
        vector.append(val)
    try:
        vector.append(count[name])
    except:
        vector.append(0)
    try:
        vector.append(sound_max.loc[name]['sound_prediction_score'])
    except:
        vector.append(0)

    vector=np.asarray(vector)
    matrix_nonan.append(vector)

matrix_nonan =np.asarray(matrix_nonan)
matrix_nonan = (matrix_nonan - np.mean(matrix_nonan, axis = 0))/(np.std(matrix_nonan, axis =0)+1e-6)
matrix_nonan = np.float32(matrix_nonan)

#print(matrix)
#print(matrix_nonan)

kernel = DotProduct() +  WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-11, 1e-1))
clf1 = GaussianProcessClassifier(kernel= kernel, random_state=0)
clf2 = svm.SVC(C =5, degree =3, probability=True, random_state = 0, kernel = 'linear', tol = 1e-3)
clf3 = RandomForestClassifier(n_estimators=3000, max_depth=6, bootstrap = True, oob_score = True, max_samples = 0.8, random_state=0)
clf4 = LGBMClassifier(n_estimators=1500, num_leaves= 2, alpha = 2, learning_rate=0.01, random_state=0)

clf1.fit(matrix_nonan, np.asarray(gs))
clf2.fit(matrix_nonan, np.asarray(gs))
clf3.fit(matrix_nonan, np.asarray(gs))
clf4.fit(matrix, np.asarray(gs))

#initial_type = [('float_input', FloatTensorType([None, matrix.shape[1]]))]
#onnx_model = onnxmltools.convert_sklearn(clf1, initial_types=initial_type, target_opset=12)
#onnxmltools.utils.save_model(onnx_model, 'model_1.onnx')

initial_type = [('float_input', FloatTensorType([None, matrix.shape[1]]))]
onnx_model = onnxmltools.convert_sklearn(clf2, initial_types=initial_type, target_opset = 15)
onnxmltools.utils.save_model(onnx_model, 'model_2.onnx')

initial_type = [('float_input', FloatTensorType([None, matrix.shape[1]]))]
onnx_model = onnxmltools.convert_sklearn(clf3, initial_types=initial_type, target_opset = 15)
onnxmltools.utils.save_model(onnx_model, 'model_3.onnx')

#initial_type = [('float_input', FloatTensorType([None, matrix.shape[1]]))]
#onnx_model = onnxmltools.convert_lightgbm(clf4, initial_types=initial_type)
#onnxmltools.utils.save_model(onnx_model, 'model_4.onnx')

df=pd.read_csv('/input/meta_info.csv')
#df = pd.read_csv('/input/meta_info.csv')
count=df['participant'].value_counts()
sound_max = df[['participant', 'sound_prediction_score']].groupby('participant').max()

FILE = open('/input/CODA_TB_Clinical_Meta_Info_Test.csv', 'r')
#FILE = open('/input/CODA_TB_Clinical_Meta_Info_Test.csv', 'r')
#FILE = open('CODA_TB_Clinical_Meta_Info_Test.csv', 'r')
l=FILE.readline()
matrix=[]
gs=[]
patient_list=[]
for l in FILE:
    l = l.strip()
    values= l.split(',')
    
    name=values.pop(0)
    patient_list.append(name)
    #label=values.pop(-1)
    vector=[]
    for val in values:
        if val == 'Male':
            val=1
        elif val == 'Female':
            val=0
        elif val == 'Yes':
            val=1
        elif val == 'No':
            val=0
        elif val == 'Not sure':
            val=np.nan
        else:
            try:
                val=float(val)
            except:
                val=np.nan
                print(val)
        vector.append(val)

    try:
        vector.append(count[name])
    except:
        vector.append(0)
    try:
        vector.append(sound_max.loc[name]['sound_prediction_score'])
    except:
        vector.append(0)
    try:
        avg=sum_month[name]/sum_num[name]
        vector.append(avg)
    except:
        print(name)
        vector.append(6)
        print('not exist')
    try:
        avg=sum_year[name]/sum_num[name]
        vector.append(avg)
    except:
        print(name)
        vector.append(2022)
        print('not exist')

    vector=np.asarray(vector)
    matrix.append(vector)

matrix=np.asarray(matrix)
matrix = (matrix - np.mean(matrix, axis = 0))/(np.std(matrix, axis =0)+1e-6)
matrix= np.float32(matrix)

matrix_nonan = []
FILE = open('/input/CODA_TB_Clinical_Meta_Info_Test.csv', 'r')
#FILE = open('/input/CODA_TB_Clinical_Meta_Info_Test.csv', 'r')
l=FILE.readline()
for l in FILE:
    l = l.strip()
    values= l.split(',')
    
    name=values.pop(0)
    #label=values.pop(-1)
    gs.append(label)
    vector=[]
    for val in values:
        if val == 'Male':
            val=1
        elif val == 'Female':
            val=0
        elif val == 'Yes':
            val=1
        elif val == 'No':
            val=0
        elif val == 'Not sure':
            val=0.5
        else:
            try:
                val=float(val)
            except:
                val= 0
                print(val)
        vector.append(val)

    try:
        vector.append(count[name])
    except:
        vector.append(0)
    try:
        vector.append(sound_max.loc[name]['sound_prediction_score'])
    except:
        vector.append(0)
    
    vector=np.asarray(vector)
    matrix_nonan.append(vector)

matrix_nonan =np.asarray(matrix_nonan)
matrix_nonan = (matrix_nonan - np.mean(matrix_nonan, axis = 0))/(np.std(matrix_nonan, axis =0)+1e-6)
matrix_nonan = np.float32(matrix_nonan)

value=(clf1.predict_proba(matrix_nonan)[:, 1]+clf2.predict_proba(matrix_nonan)[:, 1]+clf3.predict_proba(matrix_nonan)[:, 1]+clf4.predict_proba(matrix)[:, 1])/4

PRED=open('/output/predictions.csv','w')
#PRED=open('predictions.csv','w')
PRED.write('participant,probability\n')
i=0
for the_pid in patient_list:
    PRED.write(the_pid)
    PRED.write(',')
    PRED.write("%.5f" % value[i])
    PRED.write('\n')

    i=i+1
PRED.close()
