import pickle

def save_data(var,string_var):
    pickle_out = open(string_var,"wb")
    pickle.dump(var,pickle_out)
    pickle_out.close()

def load_data(string_var):
    pickle_in = open(string_var,"rb")
    var = pickle.load(pickle_in)
    pickle_in.close()
    return var