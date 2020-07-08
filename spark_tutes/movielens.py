#!/usr/bin/env python3
from pyspark import SparkContext, SparkConf

# val conf = new SparkConf()
#              .setMaster("local[2]")
#              .setAppName("ALS")
# sc = SparkContext(conf)

import time
import numpy as np

#movielens = sc.textFile("/mnt/c/Users/raj.shah/Downloads/ml-100k/ml-100k/u.data")
#clean_data = movielens.map(lambda x:x.split('\t'))

movielens = sc.textFile("/mnt/c/Users/raj.shah/Downloads/ml-20m/ml-20m/ratings.csv")
clean_data = movielens.map(lambda x:x.split(','))
first = clean_data.first()
clean_data = clean_data.filter(lambda line: line != first)

print(clean_data.first()) #u'196\t242\t3\t881250949'

print("# ratings: " + str(clean_data.count())) #100000



#Clean up the data by splitting it

#Movielens readme says the data is split by tabs and

#is user product rating timestamp





#As an example, extract just the ratings to its own RDD

#rate.first() is 3
# y[2] Movie rating by user
rate = clean_data.map(lambda y: float(y[2]))

print(rate.mean()) #Avg rating is 3.52986


wait = input("PRESS ENTER TO CONTINUE.")
#You don't have to extract data to its own RDD

#This command counts the distinct movies

#There are 1,682 movies
#y[1]: movie identifier

clean_data.map(lambda y: int(y[1])).distinct().count() 



#Extract just the users
##User number
users = clean_data.map(lambda y: int(y[0]))

users.distinct().count() #943 users

from pyspark.mllib.recommendation import ALS,MatrixFactorizationModel, Rating

ratings = clean_data.map(lambda x: Rating(int(x[0]),int(x[1]),float(x[2])))

train, test = ratings.randomSplit([0.7,0.3],7856)


#Need to cache the data to speed up training

train.cache()

test.cache()



def run_training(rank,numIterations):
	#Setting up the parameters for ALS
	print("running training rank {}, iterations {}".format(rank,numIterations))
	# rank = 20 # Latent Factors to be made

	# numIterations = 2 # Times to repeat process

	#Create the model on the training data

	model = ALS.train(train, rank=rank, iterations=numIterations)

	#ime.sleep(234)

	#Get Performance Estimate

	pred_train_input = train.map(lambda x:(x[0],x[1]))   
	pred_train = model.predictAll(pred_train_input) 


	#Organize the data to make (user, product) the key)

	true_train_reorg = train.map(lambda x:((x[0],x[1]), x[2]))
	pred_train_reorg = pred_train.map(lambda x:((x[0],x[1]), x[2]))


	train_comb = true_train_reorg.join(pred_train_reorg)
	train_mse = train_comb.map(lambda x: (x[1][0] - x[1][1])**2).mean()


	##Same for test set

	pred_test_input = test.map(lambda x:(x[0],x[1]))
	pred_test = model.predictAll(pred_test_input)

	#Organize the data to make (user, product) the key)

	true_test_reorg = test.map(lambda x:((x[0],x[1]), x[2]))
	pred_test_reorg = pred_test.map(lambda x:((x[0],x[1]), x[2]))

	test_comb = true_test_reorg.join(pred_test_reorg)
	test_mse = test_comb.map(lambda x: (x[1][0] - x[1][1])**2).mean()

	print("TRAIN TEST OUTPUT RANK: {}, nIters: {}".format(rank,numIterations))
	for i in range(10):
		print(train_comb.takeSample(0,1))
	print("End")

	return test_mse,train_mse,model

def test_model(ranks=5,iterations=10):
	# ranks = 25
	# iterations = 7

	test_mse = np.zeros((ranks,iterations))
	train_mse = np.zeros((ranks,iterations))
	from matplotlib import pyplot as plt

	for rank in range(1,ranks):
		test_itr = np.zeros(iterations)
		train_itr = np.zeros(iterations)
		itrr = np.zeros(iterations)
		for iteration in range(1,iterations):
			_test, _train, _ = run_training(rank,iteration)
			# _test = (rank**3)*iteration
			# _train = (rank**2)*iteration
			
			test_itr[iteration] = _test
			train_itr[iteration] = _train
			itrr[iteration] = iteration
		test_mse[rank] = test_itr
		train_mse[rank] = train_itr

		f = plt.figure()
		test_line = plt.plot([x for x in itrr],[x for x in test_itr],label='test')
		train_line = plt.plot([x for x in itrr],[x for x in train_itr],label='train')

		plt.title("Rank " + str(rank))
		plt.legend()
		plt.yscale('log')
		plt.show()
		f.savefig("rank_{}_.pdf".format(rank))
		plt.close(f)

test_model(5,10)
#usr_289 = train.filter(lambda x: 289 == x[0])

# test_err,train_err,model = run_training(60,20)

#true_pred.takeSample(0,1)