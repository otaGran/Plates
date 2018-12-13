import os  
import os.path  
import matplotlib.image as mpimg  
import numpy as np
import string
# this folder is custom  
rootdir="./"  
characters = string.digits + string.ascii_uppercase + "-"
batch_size,height,width,i,n_len,n_class = 11902,79,180,0,6+1,len(characters)
print(characters)
X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
for parent,dirnames,filenames in os.walk(rootdir):   
    #case 2  
    for filename in filenames:
        if filename.endswith(".jpg"):
            X[i] = mpimg.imread(filename)
            random_str = filename[filename.find('_')+1:].replace('.jpg','')
            print(random_str)
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
            i += 1    
            print("full path" + os.path.join(parent,filename)) 
 
