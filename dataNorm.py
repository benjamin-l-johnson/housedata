#!/usr/bin/env ../venv.sh

import numpy as np
import pickle
import random


def logDivFour(inp):
	return (np.log10(inp)/4)
	

def logDivTwo(inp):
	return (np.log10(inp)/2)

def logDivSix(inp):
	return (np.log10(inp)/6)

##pretty print numpy stuff##
#np.set_printoptions(precision=2,suppress=True,threshold=20)

#Make above function run on vectors/matrices (not really needed but kind of cool example)
#logDF = np.vectorize(logDivFour)


p = open("houseData.pkl","wb")
f = open("housingData.txt")

firstPass = []
for line in f:
	data = line.split()

	#convert list of strings to floats
	data = [float(x) for x in data]
	
	#add it to our list
	firstPass.append(data)

#make it a "matrix"
mat = np.array(firstPass)

#get the min and max of each column 
#cmax = np.amax(mat,axis=0)
#cmin = np.amin(mat,axis=0)
#print(cmin[5],cmax[5])


col0 = np.nan_to_num(logDivFour(mat[:,0]))

col1 = np.nan_to_num(logDivFour(mat[:,1]))

#log10(0) = -infinity this will go through and replace -inf with 0
col1[col1 <-1] = 0.0

col2 = np.nan_to_num(logDivFour(mat[:,2]))

col3 = mat[:,3]/2

col4 = mat[:,4]/2

col5 = mat[:,5] #

col6 = np.nan_to_num(logDivFour(mat[:,6]))

col7 = np.nan_to_num(logDivFour(mat[:,7]))

col8 = np.nan_to_num(logDivFour(mat[:,8]))

col9 = logDivSix(mat[:,9])

col10 = np.nan_to_num(logDivFour(mat[:,10]))

col11 = logDivSix(mat[:,11])

col12 = np.nan_to_num(logDivFour(mat[:,12]))

col13 = np.nan_to_num(logDivFour(mat[:,13]))

##writeout col4-6 to a text file
#np.savetxt("data4-6.txt",temp[:,4:6],fmt="%.2e")

#write out col5 to a different one
#np.savetxt("data5.txt",col5,fmt="%.2e")
#print("Matrix was",mat.shape)

##Store everything but the bit vector into a numpy array 5 arrays of zero for a place holder
mat = np.array([col0,col1,col2,col3,col4,np.zeros(506),np.zeros(506),np.zeros(506),np.zeros(506),np.zeros(506),col6,col7,col8,col9,col10,col11,col12,col13])

##Store a temp variable to rotate
temp = np.rot90(mat)

#print("After inserting 5 rows it is now",mat.shape)

#Rotate matrix 
mat = np.rot90(mat)

for row, value in enumerate(col5):
	#subtracting 4 since the lowest avg of rooms is 3.5 which will round up
	for column in range(int(round(value)-4)):
		mat[row][column+5] = 0.5 # 'on' state
print("#"*60)

#print(np.amax(mat[:,12],axis=0))

#np.savetxt("dataNP.txt",mat[:,4:13],fmt="%.2e")

#print(mat)
#####Debug/verification lines#####
#First row since our data was flipped upside down
#print(mat[505])

#split the last column out
#(x,y) = [mat[:,0:17],mat[:,17]]

#print the last row
#normdData = (x[-1],y[-1])
#print(normdData)

##small sample set to prove shuffle works####
# ver = mat[:5]
# print("#"*60)
# print(ver)


# random.shuffle(ver)

# #format similar to Nmist data
# (x,y) = [ver[:,0:16],ver[:,16]]

# train = (x[:2],y[:2])
# valid = (x[2:4],y[2:4])
# test = (x[4:],y[4:])

# print("#"*60,"\n")
# print(train)
# print("#"*60,"\n")
# print(valid)
# print("#"*60,"\n")
# print(test)
# print("#"*60,"\n")

##################################
# cmax = np.amax(mat,axis=0)
# cmin = np.amin(mat,axis=0)
# print(cmax,cmin)
#random.shuffle(mat)
##format similar to Nmist data
(x,y) = [mat[:,0:17],mat[:,17]]

##Print out all the data to a text file to look for errors 
#np.savetxt("dataNP.txt",mat,fmt="%-.2e")
train = (x[:280],y[:280])
valid = (x[280:435],y[280:435])
test = (x[435:],y[435:])


pickle.dump([train,valid,test],p)
p.close()
f.close()

print("\tDone you should now have a file named houseData.pkl")
print("#"*60)
