import matplotlib.pyplot as plt


with open('log.txt', 'r') as file:
    i = 0
    maximum = []
    minimum = []
    mean = []
    for row in file:
        M, m, me = [float(i) for i in row.split()]
        maximum.append(M)
        minimum.append(m)
        mean.append(me)
    plt.plot(range(len(maximum)), maximum, 'r')
    plt.plot(range(len(minimum)), minimum, 'b')
    plt.plot(range(len(mean)), mean, 'g')
    plt.show()
