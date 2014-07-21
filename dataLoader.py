
import numpy as np
import pickle


def load():
	f = open("houseData.pkl","rb")
	(ta,ta1),(v1,v2),(tr1,tr2) = pickle.load(f)
	f.close()
	
	train = [(x,[y]) for x,y in zip(ta,ta1)]
	valid = [(x,[y]) for x,y in zip(v1,v2)]
	test = [(x,[y]) for x,y in zip(tr1,tr2)]

	np.random.shuffle(train)
	np.random.shuffle(valid)
	np.random.shuffle(test)

	return train,valid,test
