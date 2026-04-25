from demos import *
import numpy as np

def main():
    print("")

    # p1_allSG(initial_pi=np.array([3, 1]))
    # p1_allSG(initial_pi=np.array([0, 0]))
    # p1_allSG(initial_pi=np.array([10, 10]))

    # p2_allSG(initial_pi=np.array([3, 1]))
    # p2_allSG(initial_pi=np.array([0, 0]))
    # p2_allSG(initial_pi=np.array([10, 10]))

    # p3_allSG("data_ocsc/wt040/wt040_001.dat")
    # p3_allSG("data_ocsc/wt040/wt040_005.dat")
    # p3_allSG("data_ocsc/wt040/wt050_001.dat")
    # p3_allSG("data_ocsc/wt040/wt050_005.dat")

    p4_allSG("data_ocsc/wt040/wt040_001.dat")
    p4_allSG("data_ocsc/wt040/wt040_005.dat")
    p4_allSG("data_ocsc/wt040/wt050_001.dat")
    p4_allSG("data_ocsc/wt040/wt050_005.dat")

    # demo_progDyn()  
    # demo_p3()  
    # demo_p4()



if __name__ == "__main__":
    main()
   
    
    