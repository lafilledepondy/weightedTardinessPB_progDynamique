from readData import *

def relax1(nbItems, T, processingTimes, dueDates, penalties):
    INF = float('inf')
    L_tab = [INF] * (T + 1)
    choice_tab = [-1] * (T + 1)
    L_tab[T] = 0

    for t in range(T-1, -1, -1):
        for i in range(nbItems):
            t_current = t + processingTimes[i]

            if t_current <= T:
                cost = max(0, penalties[i] * (t_current - dueDates[i]))
                value = cost + L_tab[t_current]

                if value < L_tab[t]:
                    L_tab[t] = value
                    choice_tab[t] = i

    print("Relaxation value:", L_tab[0])

    t = 0
    sequence = []

    while t < T:
        i = choice_tab[t]
        sequence.append(i)
        t += processingTimes[i]

    print("Sequence:", sequence)

    return L_tab, sequence 


def relax2(nbItems, T, processingTimes, dueDates, penalties):

    INF = float('inf')
    L_tab = [[INF]*(nbItems+1) for _ in range(T+1)]
    choice_tab = [[-1]*(nbItems+1) for _ in range(T+1)]
    L_tab[T][nbItems] = 0

    for t in range(T, -1, -1):
        for k in range(nbItems, -1, -1):
            if t == T and k == nbItems:
                continue
            for i in range(nbItems):
                p = processingTimes[i]
                if t + p <= T and k + 1 <= nbItems:
                    completion = t + p
                    cost = max(0, penalties[i]*(completion - dueDates[i]))
                    value = cost + L_tab[completion][k+1]

                    if value < L_tab[t][k]:
                        L_tab[t][k] = value
                        choice_tab[t][k] = i

    print("Relaxation value:", L_tab[0][0])

    t = 0
    k = 0
    sequence = []
    while t < T and k < nbItems:
        i = choice_tab[t][k]
        if i == -1:
            break
        sequence.append(i)
        t += processingTimes[i]
        k += 1

    print("Sequence:", sequence)

    return L_tab, sequence
