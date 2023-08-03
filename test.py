import tensorflow as tf 

model_path = r'D:\Desktop\Weather forecast\model\train\model.pickle'
tf.keras.models.load_model(model_path)