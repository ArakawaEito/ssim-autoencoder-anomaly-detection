import tensorflow as tf
import numpy as np
from tqdm import tqdm 


@tf.function
def Anomaly_score(x, model_x):    
    anomaly_score = tf.reduce_mean(tf.abs(x - model_x), axis=[1, 2, 3])

    return anomaly_score


def test(model, data_loader):
    """
    model         :モデル
    data_loader     :データセット->tf.data
    =========================================================
    return:
    各データごとの異常度を要素とする一次元配列(ndarray)
    """
    all_anomaly_score  =[]
    for x, y in tqdm(data_loader):
        ## 異常度の計算 ##
        model_x = model(x)
        anomaly_score = Anomaly_score(x, model_x) # -> array
        all_anomaly_score += anomaly_score.numpy().tolist()

    return np.array(all_anomaly_score)

