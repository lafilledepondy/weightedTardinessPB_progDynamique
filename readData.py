#!/usr/bin/env python
# coding: utf-8

def readData(datafilePath):
    with open(datafilePath, "r") as file: 
        line = file.readline()
        lineTab = line.split()
        nbItems = int(lineTab[0])
        processingTimes = []
        dueDates = []
        penalties = []
        for i in range(nbItems):
            line = file.readline()
            lineTab = line.split()
            processingTimes.append(int(lineTab[0]))
            dueDates.append(int(lineTab[1]))
            penalties.append(int(lineTab[2]))

    T=sum(processingTimes)
            
    # Affichage des informations lues
    print("Weighted Tardiness instance :")
    print("Number of Items : ", nbItems)
    print("Makespan : ",T)
    print("Processing times : ")
    print(processingTimes)
    print("Due dates : ")
    print(dueDates)
    print("Penalties : ")
    print(penalties)

    return nbItems, T, processingTimes, dueDates, penalties


