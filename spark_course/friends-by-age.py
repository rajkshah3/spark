from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local").setAppName("FriendsByAge")
sc = SparkContext(conf = conf)

def parseLine(line):
    fields = line.split(',')
    age = int(fields[2])
    numFriends = int(fields[3])
    return (age, numFriends)

def testoutput(x,y):
    print('x: ')
    print(x[0])
    print(x[1])

    print('y: ')
    print(y[0])
    print(y[1])
    return (x[0] + y[0], x[1] + y[1])

lines = sc.textFile("./fakefriends.csv")
rdd = lines.map(parseLine)
totalsByAge = rdd.mapValues(lambda x: (x, 1)).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
test = rdd.mapValues(lambda x: (x, 1))
test2 = test.reduceByKey(testoutput)

results = test2.collect()
for result in results:
    print(result)

# results = test2.collect()
# for result in results:
#     print(result)

exit()

# totalsByAge_mv = rdd.mapValues(lambda x: (x, 1))
# averagesByAge = totalsByAge.mapValues(lambda x: x[0] / x[1])
# results = averagesByAge.collect()
# for result in results:
#     print(result)

# results = totalsByAge_mv.collect()
# for result in results:
#     print(result)
