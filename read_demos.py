import numpy as np

file_name = './src/demo_2023-09-11-17-52-45.npy'
file = np.load(file_name,allow_pickle=True)

# print(file)
print(len(file))
for i in range(len(file)):
    print(file[i]['description'])