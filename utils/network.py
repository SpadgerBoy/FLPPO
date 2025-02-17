import random


def get_network_delay(total_clients, list):
    N = total_clients
    network_distribute = list

    clients_distribute = []
    for i in network_distribute:
        clients_distribute.append(int(i * N))

    clients_delays = []
    for i in range(clients_distribute[0]):
        sleeping_time = round(random.uniform(0.1, 1), 1)
        clients_delays.append(sleeping_time)

    for i in range(clients_distribute[1]):
        sleeping_time = round(random.uniform(1, 5), 1)
        clients_delays.append(sleeping_time)

    for i in range(clients_distribute[2]):
        if random.random() <= 0.9:
            sleeping_time = round(random.uniform(5, 10), 1)
        else:
            sleeping_time = round(random.uniform(10, 100), 1)
        clients_delays.append(sleeping_time)

    # 打乱顺序
    random.shuffle(clients_delays)

    clients_delays = [3.7, 4.5, 3.8, 8.6, 0.9, 9.5, 4.7, 3.0, 2.4, 2.4,
                      0.5, 3.6, 0.2, 1.0, 9.8, 4.4, 2.8, 0.8, 0.2, 0.1,
                      3.5, 1.4, 10.2, 3.3, 3.2, 2.3, 0.3, 4.7, 73.8, 8.6,
                      0.1, 4.7, 0.3, 1.7, 2.8, 43.2, 3.3, 0.9, 8.5, 0.5]

    print('delays:', clients_delays)
    return clients_delays


def  delay_to_prob(delay_list, group_list):
    group_delay = [[] for i in range(len(group_list))]
    weights = [[] for i in range(len(group_list))]
    for i, group in enumerate(group_list):
        for client_id in group:
            group_delay[i].append(delay_list[client_id])
            if delay_list[client_id] <= 1:
                weights[i].append(2)
            elif 1 < delay_list[client_id] <= 5:
                weights[i].append(1.5)
            elif 5 < delay_list[client_id] < 10:
                weights[i].append(1)
            else:
                weight = round(1 * 10 / delay_list[client_id], 2)
                weights[i].append(weight)

    print('1:', group_delay)
    print('2:', weights)

    new_weights = []
    for i, weights_list in enumerate(weights):
        weights_list = normalization(weights_list)
        new_weights.append(weights_list)

    new_weights = [
        [0.10344827586206896, 0.10344827586206896, 0.10344827586206896, 0.06896551724137931, 0.13793103448275862,
         0.06896551724137931, 0.10344827586206896, 0.10344827586206896, 0.10344827586206896, 0.10344827586206896],
        [0.11428571428571428, 0.08571428571428572, 0.11428571428571428, 0.11428571428571428, 0.05714285714285714,
         0.08571428571428572, 0.08571428571428572, 0.11428571428571428, 0.11428571428571428, 0.11428571428571428],
        [0.11432926829268292, 0.11432926829268292, 0.07469512195121951, 0.11432926829268292, 0.11432926829268292,
         0.11432926829268292, 0.1524390243902439, 0.11432926829268292, 0.010670731707317074, 0.07621951219512195],
        [0.13131976362442546, 0.0984898227183191, 0.13131976362442546, 0.0984898227183191, 0.0984898227183191,
         0.01510177281680893, 0.0984898227183191, 0.13131976362442546, 0.06565988181221273, 0.13131976362442546]]

    print('weights:', new_weights)

    return new_weights


def normalization(weights_list):
    t = 0
    for i, weight in enumerate(weights_list):
        t += weight

    for i, weight in enumerate(weights_list):
        weights_list[i] = weight / t

    return weights_list


if __name__ == "__main__":
    group_list = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                  [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                  [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                  [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]]

    delay_list = get_network_delay(40, [0.3, 0.5, 0.2])

    weights =  delay_to_prob(delay_list, group_list)

'''
    delay_list0 = [0.2899, 4.6608, 4.061, 2.7918, 0.1836, 3.5844, 45.8255, 0.4591, 5.1576, 1.8951,
                  3.145, 11.5189, 6.7662, 4.5368, 2.7837, 0.544, 6.0403, 1.5721, 4.331, 1.2702,
                  3.8271, 0.5253, 0.4676, 0.7707, 0.8725, 4.5868, 0.9469, 4.0206, 100, 0.4212,
                  1.6361, 4.0718, 21.1278, 0.344, 0.1349, 6.4838, 2.2819, 2.479, 2.2914, 4.722]
    weights0 = [
        [0.1358695652173913, 0.10190217391304347, 0.10190217391304347, 0.10190217391304347, 0.1358695652173913,
         0.10190217391304347, 0.014945652173913042, 0.1358695652173913, 0.06793478260869565, 0.10190217391304347],
        [0.10814708002883922, 0.06272530641672674, 0.07209805335255948, 0.10814708002883922, 0.10814708002883922,
         0.14419610670511895, 0.07209805335255948, 0.10814708002883922, 0.10814708002883922, 0.10814708002883922],
        [0.09036144578313252, 0.12048192771084336, 0.12048192771084336, 0.12048192771084336, 0.12048192771084336,
         0.09036144578313252, 0.12048192771084336, 0.09036144578313252, 0.006024096385542168, 0.12048192771084336],
        [0.10366275051831376, 0.10366275051831376, 0.03248099516240498, 0.138217000691085, 0.138217000691085,
         0.0691085003455425, 0.10366275051831376, 0.10366275051831376, 0.10366275051831376, 0.10366275051831376]]

    delay_list1 = [3.7, 4.5, 3.8, 8.6, 0.9, 9.5, 4.7, 3.0, 2.4, 2.4,
                  0.5, 3.6, 0.2, 1.0, 9.8, 4.4, 2.8, 0.8, 0.2, 0.5,
                  3.5, 1.4, 10.2, 3.3, 3.2, 2.3, 0.3, 4.7, 73.8, 8.6,
                  0.1, 4.7, 0.2, 1.7, 2.8, 43.2, 3.3, 0.9, 8.5, 0.1]

    weights1 = [[0.10344827586206896, 0.10344827586206896, 0.10344827586206896, 0.06896551724137931, 0.13793103448275862,
               0.06896551724137931, 0.10344827586206896, 0.10344827586206896, 0.10344827586206896, 0.10344827586206896],
              [0.11428571428571428, 0.08571428571428572, 0.11428571428571428, 0.11428571428571428, 0.05714285714285714,
               0.08571428571428572, 0.08571428571428572, 0.11428571428571428, 0.11428571428571428, 0.11428571428571428],
              [0.11432926829268292, 0.11432926829268292, 0.07469512195121951, 0.11432926829268292, 0.11432926829268292,
               0.11432926829268292, 0.1524390243902439, 0.11432926829268292, 0.010670731707317074, 0.07621951219512195],
              [0.13131976362442546, 0.0984898227183191, 0.13131976362442546, 0.0984898227183191, 0.0984898227183191,
               0.01510177281680893, 0.0984898227183191, 0.13131976362442546, 0.06565988181221273, 0.13131976362442546]]
'''
