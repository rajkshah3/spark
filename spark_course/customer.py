from pyspark import SparkConf, SparkContext
import re
conf = SparkConf().setMaster("local").setAppName("WordCount")
sc = SparkContext(conf = conf)

def parseLine(line):
    fields = line.split(',')
    customer_number = int(fields[0])
    item_id = int(fields[1])
    spend = float(fields[2])
    return (customer_number,item_id,spend)

# def split_words(text):
#     return re.compile(r'\W+',re.UNICODE).split(text.lower())
# def swap_key_value(x,y):
#     return (y,x)

input = sc.textFile("./customer-orders.csv")

transaction_data = input.map(parseLine)
customer_spends = transaction_data.map(lambda x: (x[0],x[2])).reduceByKey(lambda x,y : x+y).map(lambda x: (x[1],x[0])).sortByKey()

# wordCounts = words.countByValue()
# wordCounts = words.map(lambda x:(x,1)).reduceByKey(lambda x, y: x+y)
# wordCounts = wordCounts.map(swap_key_value)
# wordCounts = wordCounts.sortByKey()

results = customer_spends.collect()

for result in results:
    print(result)
    # count = str(result[0])
    # word = result[1].encode('ascii', 'ignore')
    # if (word):
    #     print(word.decode() + ":\t\t" + count)



# for word, count in wordCounts.items():
#     cleanWord = word.encode('ascii', 'ignore')
#     if (cleanWord):
#         print(cleanWord.decode() + " " + str(count))
