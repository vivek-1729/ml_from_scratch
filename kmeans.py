from matplotlib import pyplot as plt
import numpy as np
import math
from statistics import mean
x = [12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72]
y = [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24]
minimum = min([min(x), min(y)])
maximum = max([max(x), max(y)])

k=3
#TODO: If a specific centroid has no points that are closest to it, then it fails

centroids = {
    i+1: [np.random.randint(minimum, maximum), np.random.randint(minimum, maximum)]
    for i in range(k)
}
def show_points():
    plt.scatter(x,y)
    for i in centroids.values():
        plt.scatter(i[0], i[1], color='k')
    plt.show()

def euclidean_distance(x1,y1, centroid):
    x2, y2 = centroid
    distance = math.sqrt((y2-y1)**2 + (x2-x1)**2)
    return distance

def map_euclidean_distance(point):
    x1, y1 = point
    return [euclidean_distance(x1,y1,i) for i in centroids.values()]

def assign_points():
    points = [[] for _ in range(k)]
    for point in zip(x,y):
        centroid_distances = map_euclidean_distance(point)
        closest_index = centroid_distances.index(min(centroid_distances))
        points[closest_index].append(point)
    return points

def centroid_step():
    for i in centroids.values():
        plt.scatter(i[0], i[1], color='k')
    points = assign_points()
    colorArray = ['r','g', 'b']
    indices = [i+1 for i in range(k)]
    for point, color, index in zip(points, colorArray, indices):
        X,Y = list(zip(*point))
        plt.scatter(X,Y,color=color)
        centroids[index] = [mean(X), mean(Y)]
    plt.show()

for i in range(5):
    show_points()
    assign_points()
    centroid_step()
