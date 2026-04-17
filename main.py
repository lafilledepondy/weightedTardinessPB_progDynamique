from simpleSubgradient import *

def main():
    print("")
    
    # print(basic_subgradient([3,1], 0, 0.000001))

    print(subgradientPolyak([3,1], 0, 0.000001)[0])

if __name__ == "__main__":
    main()