
import pickle
import numpy as np
import lightgbm as lgb
import onnxruntime as rt
import time

FILE=open('/input/meta_info.csv','r')
FILE.readline()
test_data=[]
for lll in FILE:
    lll=lll.strip()
    ttt=lll.split(',')
    data=np.load('set_final/' + ttt[1]+ '.npy')
    ttt=ttt[1].split('-recording')
    my_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(float(ttt[0])/1000)))
    t=my_time.split(' ')
    t=t[0].split('-')
    data=np.hstack(((float(t[0])),data))
    data=np.hstack(((float(t[1])),data))
    test_data.append(data)
FILE.close()

test_feature=np.asarray(test_data)
test_feature=test_feature.astype('float32')
model = pickle.load(open('model_final.pkl', "rb"))
pred=model.predict(test_feature)


pred=(pred+0.25)/2


FILE=open('/input/meta_info.csv','r')
pred_final={}
count_final={}
FILE.readline()
iii=0
for lll in FILE:
    lll=lll.strip()
    ttt=lll.split(',')
    if (ttt[0] in pred_final):
        pred_final[ttt[0]]=pred[iii]+pred_final[ttt[0]]
        count_final[ttt[0]]=count_final[ttt[0]]+1
    else:
        pred_final[ttt[0]]=pred[iii]
        count_final[ttt[0]]=1

    iii=iii+1

NEW = open('/output/predictions.csv', 'w')
NEW.write('participant,probability\n')
for patient in pred_final.keys():
    average = pred_final[patient]/count_final[patient]
    NEW.write(str(patient))
    NEW.write(',')
    NEW.write("%.6f" % (average))
    NEW.write('\n')
NEW.close()

