import pytest

from readData import *
from progDyn import *
from main import optimalOrRealisableOrInfesable

# ====== Relax. Lin. 1 
def test_RL1_Toy_wt4_1_dat():
    datafilePath = 'data/Toy_wt4.1.dat'
    nbItems, T, processingTimes, dueDates, penalties = readData(datafilePath)
    L_tab, sequence = relax1(nbItems, T, processingTimes, dueDates, penalties)
    status = optimalOrRealisableOrInfesable(sequence, nbItems, T, processingTimes)

    assert L_tab[0] == 67
    assert sequence == [0, 1, 3, 2]
    assert status == "Solution réalisable (donc optimale)"

def test_RL1_Toy_wt4_2_dat():
    datafilePath = 'data/Toy_wt4.2.dat'
    nbItems, T, processingTimes, dueDates, penalties = readData(datafilePath)
    L_tab, sequence = relax1(nbItems, T, processingTimes, dueDates, penalties)
    status = optimalOrRealisableOrInfesable(sequence, nbItems, T, processingTimes)

    assert L_tab[0] == 42
    assert sequence == [0, 1, 1, 1, 1, 2]
    assert status == "Solution NON réalisable (borne dual)"        

def test_RL1_wt040_wt040_001_dat():
    datafilePath = 'data/wt040/wt040_001.dat'
    nbItems, T, processingTimes, dueDates, penalties = readData(datafilePath)
    L_tab, sequence = relax1(nbItems, T, processingTimes, dueDates, penalties)
    status = optimalOrRealisableOrInfesable(sequence, nbItems, T, processingTimes)

    assert L_tab[0] == 396
    assert len(sequence) == 71
    assert status == "Solution NON réalisable (borne dual)"      
    
def test_RL1_wt040_wt040_002_dat():
    datafilePath = 'data/wt040/wt040_002.dat'
    nbItems, T, processingTimes, dueDates, penalties = readData(datafilePath)
    L_tab, sequence = relax1(nbItems, T, processingTimes, dueDates, penalties)
    status = optimalOrRealisableOrInfesable(sequence, nbItems, T, processingTimes)

    assert L_tab[0] == 545
    assert len(sequence) == 39
    assert status == "Solution NON réalisable (borne dual)"     

# ====== Relax. Lin. 2
def test_RL2_Toy_wt4_1_dat():
    datafilePath = 'data/Toy_wt4.1.dat'
    nbItems, T, processingTimes, dueDates, penalties = readData(datafilePath)
    L_tab, sequence = relax2(nbItems, T, processingTimes, dueDates, penalties)
    status = optimalOrRealisableOrInfesable(sequence, nbItems, T, processingTimes)

    assert L_tab[0][0] == 67
    assert sequence == [0, 1, 3, 2]
    assert status == "Solution réalisable (donc optimale)"    

def test_RL2_Toy_wt4_2_dat():
    datafilePath = 'data/Toy_wt4.2.dat'
    nbItems, T, processingTimes, dueDates, penalties = readData(datafilePath)
    L_tab, sequence = relax2(nbItems, T, processingTimes, dueDates, penalties)
    status = optimalOrRealisableOrInfesable(sequence, nbItems, T, processingTimes)

    assert L_tab[0][0] == 42
    assert sequence == [0, 1, 3, 2]
    assert status == "Solution réalisable (donc optimale)"    

def test_RL2_wt040_wt040_001_dat():
    datafilePath = 'data/wt040/wt040_001.dat'
    nbItems, T, processingTimes, dueDates, penalties = readData(datafilePath)
    L_tab, sequence = relax2(nbItems, T, processingTimes, dueDates, penalties)
    status = optimalOrRealisableOrInfesable(sequence, nbItems, T, processingTimes)

    assert L_tab[0][0] == 396
    assert len(sequence) == 40
    assert status == "Solution NON réalisable (borne dual)"      
    
def test_RL2_wt040_wt040_002_dat():
    datafilePath = 'data/wt040/wt040_002.dat'
    nbItems, T, processingTimes, dueDates, penalties = readData(datafilePath)
    L_tab, sequence = relax2(nbItems, T, processingTimes, dueDates, penalties)
    status = optimalOrRealisableOrInfesable(sequence, nbItems, T, processingTimes)

    assert L_tab[0][0] == 545
    assert len(sequence) == 40
    assert status == "Solution NON réalisable (borne dual)"      