
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRegressor
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
import onnxmltools
import pickle
FILE=open('CODA_TB_Solicited_Meta_Info.csv','r')
train_data=[]
train_gs=[]
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


train_data=np.asarray(train_data).astype('float32')
train_gs=np.asarray(train_gs).astype('float32')

lgb_train = lgb.Dataset(train_data, train_gs)

params = {
    'boosting_type': 'gbdt',
    'n_estimators': 1000,
    'objective': 'regression',
    'learning_rate': 0.04,
    'reg_alpha': 2.0,
}

model= lgb.train(params,
    lgb_train,
    num_boost_round=1000
#    num_boost_round=1
)
pickle.dump(model, open('model_final.pkl', 'wb'))

initial_type = [('float_input', FloatTensorType([None, train_data.shape[1]]))]
onnx_model = onnxmltools.convert_lightgbm(model, initial_types=initial_type)
onnxmltools.utils.save_model(onnx_model, 'model_final.onnx')
