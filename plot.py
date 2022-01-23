import matplotlib.pyplot as plt


with open('log.txt', 'r') as file:
    i = 0
    maximum = []
    minimum = []
    for row in file:
        M, m = [int(i) for i in row.split()]
        maximum.append(M)
        minimum.append(m)
    plt.plot(range(len(maximum)), maximum, 'r')
    plt.plot(range(len(minimum)), minimum, 'b')
    plt.show()
