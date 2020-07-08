from pyspark import SparkConf, SparkContext
import re
conf = SparkConf().setMaster("local").setAppName("WordCount")
sc = SparkContext(conf = conf)

def split_words(text):
    return re.compile(r'\W+',re.UNICODE).split(text.lower())
def swap_key_value(x,y):
    return (y,x)
input = sc.textFile("./book.txt")

words = input.flatMap(split_words)
# wordCounts = words.countByValue()
wordCounts = words.map(lambda x:(x,1)).reduceByKey(lambda x, y: x+y)
# wordCounts = wordCounts.map(swap_key_value)
# wordCounts = wordCounts.sortByKey()

results = wordCounts.collect()

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
