from sklearn.cluster import KMeans


class kmeans_groups:
    def __init__(self, x, n_parts, length):
        self.arr = x
        self.n = n_parts
        self.k = length

    def split(self, lis):
        arr = []

        for i in lis:
            arr.append(self.arr[i])

        n_clusters = round(len(lis) / (self.k * 2))

        kmeans = KMeans(n_clusters=n_clusters).fit(arr)

        labels = kmeans.labels_

        new_lis = [[] for i in range(n_clusters)]
        for i in range(len(labels)):
            new_lis[labels[i]].append(lis[i])

        return new_lis

    def merge(self, lis):

        new_lis = []
        index = []

        if len(lis) == 1:
            new_lis.append(lis[0])
            return new_lis

        for i in range(len(lis)):
            temp = []
            if i not in index:
                gap = float('inf')
                for j in range(i + 1, len(lis)):
                    if j not in index:
                        gap1 = len(lis[i]) + len(lis[j]) - self.k
                        if gap > abs(gap1) or (gap == abs(gap1) and gap1 < 0):
                            gap = abs(gap1)
                            temp = [gap1, i, j]
                new_lis.append(lis[temp[1]] + lis[temp[2]])
                index.append(temp[1])
                index.append(temp[2])
        return new_lis

    def groups(self, labels):

        lis = [[] for i in range(self.n)]
        for i in range(len(labels)):
            lis[labels[i]].append(i)

        print(f'old_groups: {lis}')

        k = self.k
        new_list = []
        merge_list = []
        split_list = []
        for i in range(len(lis)):
            if len(lis[i]) < k:  # 小的组合并
                merge_list.append(lis[i])
            elif k <= len(lis[i]) <= k * 2:
                new_list.append(lis[i])
            else:  # 大的组拆分
                print(f'old_split_list: {lis[i]}')
                split_list = self.split(lis[i])
                print(f'new_split_list: {split_list}')
                for j in split_list:
                    new_list.append(j)
        merge_list = sorted(merge_list, key=lambda x: len(x))

        print(f'merge_list: {merge_list}')

        merge_list = self.merge(merge_list)

        for i in range(len(merge_list)):
            new_list.append(merge_list[i])

        return new_list

    def kmeans_clusters(self):
        '''
        kmeans = KMeans(n_clusters=self.n).fit(self.arr)

        labels = kmeans.labels_

        group_list = self.groups(labels)
        '''

        '''group_list = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                      [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                      [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
                      [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
                      [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
                      [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
                      [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
                      [80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
                      [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]]'''
        group_list = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                      [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                      [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]]

        group_list = [ [] for i in range(int(len(self.arr)/10)) ]
        for i in range(int(len(self.arr)/10)):
            for j in range(10):
                group_list[i].append(i*10+j)
        return group_list


'''
labels = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
          1,1,1,1,1,1,1,
          2,
          3,3,3,
          4,4,4,
          5,5,5,5,
          6,6,
          7,
          8,8,8,
          9,9]

g = kmeans_groups([], 10, 5)

list = g.groups(labels)
print(list)
'''
