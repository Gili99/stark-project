import numpy as np
import math

class GeometricalEstimation(object):
    def __init__(self):
        self.name = 'Geometrical estimation'

    def euclideanDist(self, pointA, pointB):
        return math.sqrt((pointA[0] - pointB[0]) ** 2 + (pointA[1] - pointB[1]) ** 2)

    def calculate_geo_estimation(self, channelsAtTime, coordinates):
        maxVal = channelsAtTime.max()
        channelsAtTime = channelsAtTime / maxVal
        geoX = sum([coordinates[i][0] * channelsAtTime[i] for i in range(8)])
        geoY = sum([coordinates[i][1] * channelsAtTime[i] for i in range(8)])
        return (geoX, geoY)

    def calculate_shifts(self, geoAvgs):
        shifts = np.zeros((1, 31))
        for i in range(1, 32):
            shifts[0][i-1] = self.euclideanDist((geoAvgs[i-1][0], geoAvgs[i-1][1]), (geoAvgs[i][0], geoAvgs[i][1]))
        return shifts
    
    def calculateFeature(self, spikeList):
        result = np.zeros((len(spikeList), 2))
        coordinates = [(0, 0), (-9, 20), (8, 40), (-13, 60), (12, 80), (-17, 100), (16, 120), (-21, 140)]

        for j, spike in enumerate(spikeList):
            geoAvgs = np.zeros((32, 2))

            # Turn the arr into absolute values
            absolute = lambda x: abs(x)
            vfunc = np.vectorize(absolute)
            absArr = vfunc(spike.get_data())

            for i in range(32):
                channels = absArr[:, i]
                geoAvgs[i, 0], geoAvgs[i, 1] = self.calculate_geo_estimation(channels, coordinates)

            shifts = self.calculate_shifts(geoAvgs)

            result[j, 0] = np.mean(shifts, axis = 1)
            result[j, 1] = np.std(shifts, axis = 1)
        return result

    def get_headers(self):
        return ["geometrical_avg_shift", "geometrical_shift_sd"]
