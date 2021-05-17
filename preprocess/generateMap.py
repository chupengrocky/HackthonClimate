import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd



class MapGenerator:

    def __init__(self, moduelNum, dataSource="../data_set/present_data.csv"):
        self.num = moduelNum
        self.source = dataSource


    def dataLoader(self):
        dataf = pd.read_csv(self.source)
        
        return None

    def showGrid(self):
        return None
    
M = MapGenerator(10)

