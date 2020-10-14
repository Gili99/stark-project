import numpy as np
import math

class GeometricalEstimation(object):
    def __init__(self):
        self.name = 'Geometrical estimation'

    def euclideanDist(self, pointA, pointB):
        return math.sqrt((pointA[0] - pointB[0]) ** 2 + (pointA[1] - pointB[1]) ** 2)

    def calculate_geo_estimation(self, channelsAtTime, coordinates):
        maxVal = channelsAtTime.max()
        
        total = 0
        for i in range(8):
            entry = channelsAtTime[i]
            if entry < 0:
                entry *= -1
            total += entry
        channelsAtTime = channelsAtTime / total
        geoX = sum([coordinates[i][0] * channelsAtTime[i] for i in range(8)])
        geoY = sum([coordinates[i][1] * channelsAtTime[i] for i in range(8)])
        return (geoX, geoY)

    def calculate_shifts_2D(self, geoAvgs):
        shifts = np.zeros((1, 31))
        for i in range(1, 32):
            shifts[0][i - 1] = self.euclideanDist((geoAvgs[i - 1][0], geoAvgs[i-1][1]), (geoAvgs[i][0], geoAvgs[i][1]))
        return shifts

    def calculate_shifts_1D(self, geoAvgs, d):
        shifts = np.zeros((1, 31))
        for i in range(1, 32):
            shifts[0][i - 1] = geoAvgs[i][d] - geoAvgs[i-1][d]
        return shifts

    def calc_max_dist(self, cordinates):
        max_dist = 0
        for i, cor1 in enumerate(cordinates[:-1]):
            for cor2 in cordinates[i + 1:]:
                dist = self.euclideanDist((cor1[0], cor1[1]), (cor2[0], cor2[1]))
                if dist > max_dist:
                    max_dist = dist
        return max_dist
    
    def calculateFeature(self, spikeList):
        result = np.zeros((len(spikeList), 4))
        coordinates = [(0, 0), (-9, 20), (8, 40), (-13, 60), (12, 80), (-17, 100), (16, 120), (-21, 140)]

        for j, spike in enumerate(spikeList):
            geoAvgs = np.zeros((32, 2))

            arr = spike.get_data()
            for i in range(32):
                channels = arr[:, i] * (-1)
                geoAvgs[i, 0], geoAvgs[i, 1] = self.calculate_geo_estimation(channels, coordinates)

            shifts_2D = self.calculate_shifts_2D(geoAvgs)

            result[j, 0] = np.mean(shifts_2D, axis = 1)
            result[j, 1] = np.std(shifts_2D, axis = 1)
            result[j, 2] = self.euclideanDist((geoAvgs[0][0], geoAvgs[0][1]), (geoAvgs[-1][0], geoAvgs[-1][1]))
            result[j, 3] = self.calc_max_dist(geoAvgs)
        return result

    def get_headers(self):
        return ["geometrical_avg_shift", "geometrical_shift_sd", "geometrical_displacement", "geometrical_max_dist"]
