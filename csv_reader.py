import csv
import numpy as np
def RoadPoints(filename):
    with open(filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        return_array=[]
        for i,row in enumerate(spamreader):
            return_array.append([float(row[0][2:]),float(row[1][2:]),float(row[2][2:])])
        np_points=np.array(return_array)
    return np_points

