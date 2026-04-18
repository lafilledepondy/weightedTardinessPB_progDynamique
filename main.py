from simpleSubgradient import subgradient_basic, subgradient_Polyak, subgradient_ADS

def main():
    print("")
    
    # print(subgradient_basic([3,1], 0, 0.000001))

    print(subgradient_Polyak([3,1], 0, 0.000001)[0])

if __name__ == "__main__":
    main()