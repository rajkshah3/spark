import sys
from pyspark import SparkConf, SparkContext
from math import sqrt
import numpy as np
def loadMovieNames():
    movieNames = {}
    movieGenres = {}
    with open("./ml-100k/u.ITEM", encoding='ascii', errors='ignore') as f:
        for line in f:
            fields = line.replace('\n','').split('|')
            movieNames[int(fields[0])] = fields[1]
            genres = [ int(i) for i in fields[5:]]
            movieGenres[int(fields[0])] = genres
    return movieNames, movieGenres

#Python 3 doesn't let you pass around unpacked tuples,
#so we explicitly extract the ratings now.
def makePairs( userRatings ):
    ratings = userRatings[1]
    (movie1, rating1) = ratings[0]
    (movie2, rating2) = ratings[1]
    return ((movie1, movie2), (rating1, rating2))

def shared_genre(m1,m2):
    # print(m1)
    # print(m2)
    return bool(np.dot(m1,m2))


def filterDuplicates( userRatings ):
    ratings = userRatings[1]
    (movie1, rating1) = ratings[0]
    (movie2, rating2) = ratings[1]
    return movie1 < movie2

def computeCosineSimilarity(ratingPairs):
    numPairs = 0
    sum_xx = sum_yy = sum_xy = 0
    for ratingX, ratingY in ratingPairs:
        sum_xx += ratingX * ratingX
        sum_yy += ratingY * ratingY
        sum_xy += ratingX * ratingY
        numPairs += 1

    numerator = sum_xy
    denominator = sqrt(sum_xx) * sqrt(sum_yy)

    score = 0
    if (denominator):
        score = (numerator / (float(denominator)))

    return (score, numPairs)


conf = SparkConf().setMaster("local[*]").set("spark.driver.host",'localhost').setAppName("MovieSimilarities")
sc = SparkContext(conf = conf)

print("\nLoading movie names...")
nameDict, genre_dict = loadMovieNames()
# print(genre_dict)
data = sc.textFile("./ml-100k/u.data")


# Map ratings to key / value pairs: user ID => movie ID, rating
ratings = data.map(lambda l: l.split()).map(lambda l: (int(l[0]), (int(l[1]), float(l[2]))))

# r = ratings.take(10)
# for rating in r:
#     print(rating)

#Convert to average centered ratings
only_ratings = ratings.map(lambda l: (l[1][0], l[1][1]))
only_ratings = only_ratings.groupByKey().mapValues(lambda x: np.sum([i for i in x])/len([i for i in x]))
# orr = only_ratings.take(10)
# print(orr)
# exit()
ratings_dict = { i:j for i,j in only_ratings.collect()}
# print('ratings dict:')
# print(ratings_dict)
average_rating_dict = sc.broadcast(ratings_dict)

ratings = ratings.mapValues(lambda l: (l[0], l[1] - average_rating_dict.value[l[0]]))


def CosineSimilarityWithGenreScale(line):
    numPairs = 0
    sum_xx = sum_yy = sum_xy = 0
    movie1 = line[0][0]
    movie2 = line[0][1]
    ratingPairs = line[1]
    for ratingX, ratingY in ratingPairs:
        sum_xx += ratingX * ratingX
        sum_yy += ratingY * ratingY
        sum_xy += ratingX * ratingY
        numPairs += 1

    numerator = sum_xy
    denominator = sqrt(sum_xx) * sqrt(sum_yy)
    genre_metric = np.dot(genre_dict[movie1],genre_dict[movie2])/(np.sum(genre_dict[movie1]) + np.sum(genre_dict[movie2]))

    score = 0
    if (denominator):
        score = genre_metric * (numerator / (float(denominator)))

    return ((movie1,movie2),(score, numPairs))

def CosineSimilarityWithCoRatings(line):
    numPairs = 0
    sum_xx = sum_yy = sum_xy = 0
    movie1 = line[0][0]
    movie2 = line[0][1]
    ratingPairs = line[1]
    for ratingX, ratingY in ratingPairs:
        sum_xx += ratingX * ratingX
        sum_yy += ratingY * ratingY
        sum_xy += ratingX * ratingY
        numPairs += 1

    numerator = sum_xy
    denominator = sqrt(sum_xx) * sqrt(sum_yy)
    ratings_metric = np.log(len([i for i in ratingPairs])/10)
    score = 0
    if (denominator):
        score = ratings_metric * (numerator / (float(denominator)))

    return ((movie1,movie2),(score, numPairs))

# Emit every movie rated together by the same user.
# Self-join to find every combination.
joinedRatings = ratings.join(ratings)

# At this point our RDD consists of userID => ((movieID, rating), (movieID, rating))

# Filter out duplicate pairs
uniqueJoinedRatings = joinedRatings.filter(filterDuplicates)

# Now key by (movie1, movie2) pairs.
moviePairs = uniqueJoinedRatings.map(makePairs)

# We now have (movie1, movie2) => (rating1, rating2)
# Now collect all ratings for each movie pair and compute similarity
moviePairRatings = moviePairs.groupByKey()

# We now have (movie1, movie2) = > (rating1, rating2), (rating1, rating2) ...
# Can now compute similarities.
# moviePairSimilarities = moviePairRatings.mapValues(computeCosineSimilarity).cache()
# print(moviePairRatings.take(10))
# exit()
moviePairSimilarities = moviePairRatings.map(CosineSimilarityWithCoRatings).cache()

# Save the results if desired
#moviePairSimilarities.sortByKey()
#moviePairSimilarities.saveAsTextFile("movie-sims")

def result_filter(pairSim):
    t1 = (pairSim[0][0] == movieID)
    t2 = (pairSim[0][1] == movieID)
    t11 = False
    if(t1):
        t11 = shared_genre(genre_dict[pairSim[0][1]],genre_dict[movieID])
    if(t2):
        t11 = shared_genre(genre_dict[pairSim[0][0]],genre_dict[movieID])

    t3 = pairSim[1][0] > scoreThreshold 
    t4 = pairSim[1][1] > coOccurenceThreshold
        #Average rating threshold
    t5 = average_rating_dict.value[movieID] > average_rating_threshold
        #At least one shared genre threshold
    if((t1 or t2) and t11 and t3 and t4 and t5):
        return True

    return False
# Extract similarities for the movie we care about that are "good".
if (len(sys.argv) > 1):

    scoreThreshold = 0.01
    coOccurenceThreshold = 25
    average_rating_threshold = 2
    movieID = int(sys.argv[1])

    # Filter for movies with this sim that are "good" as defined by
    # our quality thresholds above
    # filteredResults = moviePairSimilarities.filter(lambda pairSim: \
    #     (pairSim[0][0] == movieID or pairSim[0][1] == movieID) \
    #     and pairSim[1][0] > scoreThreshold and pairSim[1][1] > coOccurenceThreshold)

    filteredResults = moviePairSimilarities.filter(result_filter) 

    # Sort by quality score.
    results = filteredResults.map(lambda pairSim: (pairSim[1], pairSim[0])).sortByKey(ascending = False).take(10)

    print("Top 10 similar movies for " + nameDict[movieID])
    # print(genre_dict[movieID])
    for result in results:
        (sim, pair) = result
        # Display the similarity result that isn't the movie we're looking at
        similarMovieID = pair[0]
        if (similarMovieID == movieID):
            similarMovieID = pair[1]
        print(nameDict[similarMovieID] + "\tscore: " + str(sim[0]) + "\tstrength: " + str(sim[1]))
        # print(genre_dict[similarMovieID])
