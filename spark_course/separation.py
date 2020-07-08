from pyspark import SparkConf, SparkContext
import re
conf = SparkConf().setMaster("local").set("spark.driver.host",'localhost').setAppName("DegreesOfSeparation")
sc = SparkContext(conf = conf)

startCharacterID = 5306 #SpiderMan
targetCharacterID = 14  #ADAM 3,031 (who?)

# Our accumulator, used to signal when we find the target character during
# our BFS traversal.
hitCounter = sc.accumulator(0)


def bfsReduce(data1, data2):
    edges1 = data1[0]
    edges2 = data2[0]
    distance1 = data1[1]
    distance2 = data2[1]
    color1 = data1[2]
    color2 = data2[2]

    distance = 9999
    color = color1
    edges = []

    # See if one is the original node with its connections.
    # If so preserve them.
    if (len(edges1) > 0):
        edges.extend(edges1)
    if (len(edges2) > 0):
        edges.extend(edges2)

    # Preserve minimum distance
    if (distance1 < distance):
        distance = distance1

    if (distance2 < distance):
        distance = distance2

    # Preserve darkest color
    if (color1 == 'White' and (color2 == 'Grey' or color2 == 'Black')):
        color = color2

    if (color1 == 'Grey' and color2 == 'Black'):
        color = color2

    if (color2 == 'White' and (color1 == 'Grey' or color1 == 'Black')):
        color = color1

    if (color2 == 'Grey' and color1 == 'Black'):
        color = color1

    return (edges, distance, color)


def bfsMap(node):
    characterID = node[0]
    data = node[1]
    connections = data[0]
    distance = data[1]
    color = data[2]

    results = []

    #If this node needs to be expanded...
    if (color == 'Grey'):
        for connection in connections:
            newCharacterID = connection
            newDistance = distance + 1
            newColor = 'Grey'
            if (targetCharacterID == connection):
                hitCounter.add(1)

            newEntry = (newCharacterID, ([], newDistance, newColor))
            results.append(newEntry)

        #We've processed this node, so color it Black
        color = 'Black'

    #Emit the input node so we don't lose it.
    results.append( (characterID, (connections, distance, color)) )
    return results


def convertToBFS(line):
    fields = line.split()
    heroID = int(fields[0])
    connections = [ int(con) for con in fields[1:] ]
    colour = 'White'
    distance = 9999
    if(heroID==startCharacterID):
        colour = 'Grey'
    distance = 0
    return (heroID, (connections,distance, colour))


data = sc.textFile("./Marvel-Graph.txt")
data = data.map(convertToBFS)

# words = input.flatMap(split_words)
# wordCounts = words.countByValue()
# wordCounts = words.map(lambda x:(x,1)).reduceByKey(lambda x, y: x+y)
# wordCounts = wordCounts.map(swap_key_value)
# wordCounts = wordCounts.sortByKey()

# results = wordCounts.collect()

for iteration in range(0, 10):
    print("Running BFS iteration# " + str(iteration+1))

    # Create new vertices as needed to darken or reduce distances in the
    # reduce stage. If we encounter the node we're looking for as a Grey
    # node, increment our accumulator to signal that we're done.
    mapped = data.flatMap(bfsMap)

    # Note that mapped.count() action here forces the RDD to be evaluated, and
    # that's the only reason our accumulator is actually updated.

    print("Processing " + str(mapped.count()) + " values.")
    print(hitCounter.value)
    if (hitCounter.value > 0):
        print("Hit the target character! From " + str(hitCounter.value) \
            + " different direction(s).")
        break

    # Reducer combines data for each character ID, preserving the darkest
    # color and shortest path.
    data = mapped.reduceByKey(bfsReduce)
# for result in results:
#     print(result)
    # count = str(result[0])
    # word = result[1].encode('ascii', 'ignore')
    # if (word):
    #     print(word.decode() + ":\t\t" + count)



# for word, count in wordCounts.items():
#     cleanWord = word.encode('ascii', 'ignore')
#     if (cleanWord):
#         print(cleanWord.decode() + " " + str(count))
