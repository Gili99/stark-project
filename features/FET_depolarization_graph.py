import numpy as np
import math

class Graph(object):
    def __init__(self, startNodes, endNodes, graphMatrix):
        self.startNodes = startNodes
        self.endNodes = endNodes
        self.graphMatrix = graphMatrix

    def averageWeight(self):
        total = 0
        counter = 0
        for i in range(8):
            for j in range(8):
                if self.graphMatrix[i, j] != -1:
                    total += self.graphMatrix[i, j]
                    counter +=1
        if counter == 0:
            return 0
        return (total / counter)

    def _getAllEdges(self):
        edges = []
        for i in range(8):
            for j in range(8):
                if self.graphMatrix[i, j] != -1:
                    edges.append((i, j))
        return edges

    def _belmanFord(self, srcNode):
        sizeV = 8 # this is |V|
        dists = [float('inf') for i in range(sizeV)]
        dists[srcNode] = 0

        for i in range(sizeV - 1):
            edges = self._getAllEdges()
            for edge in edges:
                u = edge[0]
                v = edge[1]
                weight = self.graphMatrix[u, v]
                if dists[v] > dists[u] + weight:
                    dists[v] = dists[u] + weight

        return dists

    def _findMinimumDistToEndNodes(self, dists):
        minimum = float('inf')
        for endNode in self.endNodes:
            if dists[endNode] < minimum:
                minimum = dists[endNode]
        return minimum

    def shortestDistanceFromSrcToEnd(self):
        totalMinDist = float('inf')
        for srcNode in self.startNodes:
            shortDists = self._belmanFord(srcNode)
            minDist = self._findMinimumDistToEndNodes(shortDists)
            if minDist < totalMinDist:
                totalMinDist = minDist
        return totalMinDist

class DepolarizationGraph(object):
    def __init__(self):
        self.name = 'depolarization graph'

    def euclideanDist(self, pointA, pointB):
        return math.sqrt((pointA[0] - pointB[0]) ** 2 + (pointA[1] - pointB[1]) ** 2)

    def calculateDistancesMatrix(self, coordinates):
        distances = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                distances[i, j] = self.euclideanDist(coordinates[i], coordinates[j])

        return distances

    def get_indices_with_one(self, arr):
        lst = []
        for i in range(len(arr)):
            if arr[i] == 1:
                lst.append(i)
        return lst
    
    def calculateFeature(self, spikeList):
        coordinates = [(0, 0), (-9, 20), (8, 40), (-13, 60), (-12, 80), (-17, 100), (16, 120), (-21, 140)]
        dists = self.calculateDistancesMatrix(coordinates)
        result = np.zeros((len(spikeList), 2))

        for index, spike in enumerate(spikeList):
            arr = spike.data
            minVal = arr.min()
            threshold = 0.3 * minVal

            # Detrmine where the maximum depolarization resides wrt each channel (that surpasses the threshold)
            depolarizationStatus = np.zeros((8, 32))
            for i in range(8):
                maxDepIndex = arr[i].argmin()
                if arr[i, maxDepIndex] <= threshold:
                    depolarizationStatus[i, maxDepIndex] = 1

            # Find the channels that have reached max depolarization in each timestamp
            ds = depolarizationStatus
            gTemp = []
            for j in range(32):
                indices = self.get_indices_with_one(ds[:, j])
                if len(indices) > 0:
                    gTemp.append((j, indices))

            # Build the actual graph
            graphMatrix = np.ones((8, 8)) * (-1)
            startNodes = gTemp[0][1]
            endNodes = gTemp[len(gTemp)-1][1]
            for i in range(len(gTemp)-1):
                # each entry in gTemp is of the form (timestamp, list of indices)
                fromTimestamp = gTemp[i][0]
                for fromNode in gTemp[i][1]:
                    toTimestamp = gTemp[i+1][0]
                    for toNode in gTemp[i+1][1]:
                        #print(str(fromNode) + " -> " + str(toNode) + " With weight: " + str(distances[fromNode, toNode]))
                        velocity = dists[fromNode, toNode] / (toTimestamp - fromTimestamp)
                        graphMatrix[fromNode][toNode] = velocity

            graph = Graph(startNodes, endNodes, graphMatrix)
            
            # Calculate features from the graph
            result[index, 0] = graph.averageWeight()
            result[index, 1] = graph.shortestDistanceFromSrcToEnd()
        return result

    def get_headers(self):
        return ["graph_avg_speed", "graph_shortest_distance"]

if __name__ == "__main__":
    f = DepolarizationGraph()
    mat = np.ones((8, 8)) * -1
    mat[0, 1] = 5
    mat[1, 2] = 2
    mat[0, 2] = 3
    G = Graph([0], [2], mat)
    print(G.shortestDistanceFromSrcToEnd())