import numpy as np
import pickle
import matplotlib.pyplot as plt


# Get the pickled data for one rat
with open("./dataset/data_files/26148_020520_bank0_s1_light.pkl", "rb") as file:
    data = pickle.load(file)


print(type(data))
print(data.keys())


