import numpy as np
import pickle
from tqdm import tqdm


pickle_in = open("X_train","rb")
X_train = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open("Y_train","rb")
Y_train = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open("X_valid","rb")
X_valid = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open("Y_valid","rb")
Y_valid = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open("X_test","rb")
X_test = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open("Y_test","rb")
Y_test = pickle.load(pickle_in)
pickle_in.close()

#preprocessing : normalization of data

X_train_reshape = X_train.reshape(X_train.shape[0],150,150,3) / 255
X_valid_reshape = X_valid.reshape(X_valid.shape[0],150,150,3) / 255
X_test_reshape = X_test.reshape(X_test.shape[0],150,150,3) / 255

pickle_out = open("X_train_reshape","wb")
pickle.dump(X_train_reshape,pickle_out)
pickle_out.close()

pickle_out = open("X_valid_reshape","wb")
pickle.dump(X_valid_reshape,pickle_out)
pickle_out.close()

pickle_out = open("X_test_reshape","wb")
pickle.dump(X_test_reshape,pickle_out)
pickle_out.close()



