from algs.link_infer_alg import Algorithm
import numpy as np

class CompressSensing(Algorithm):
    def __init__(self):
        super(CompressSensing, self).__init__()
        # 所有路劲的集合
        self.__paths = set()
        # 拥塞路劲的集合
        self.__congested_paths = set()
        # 不拥塞路劲的集合
        self.__un_congested_paths = set()
        # 所有链路的集合
        self.__links = set()
        # 由已知条件能够推断出的一定不拥塞链路的集合
        self.__un_congested_links = set()
        # 排除所有不可能拥塞的链路，剩下的链路就是可疑的拥塞链路。
        self.__suspected_congested_links = set()

        self.__solution = set()
        self.QB = set()

    def diagnose(self):
        # 获取拥塞路劲的集合
        self.get_paths()
        # print("拥塞路劲：", self.__congested_paths)
        # print("不拥塞的路劲：", self.__un_congested_paths)
        self.get_links()
        # print("所有链路：", self.__links)
        # print("不拥塞的链路：", self.__un_congested_links)
        # print("可以链路：", self.__suspected_congested_links)

        self.QB = self.__congested_paths
        # self.link_state_inferred = np.zeros(self.link_num)
        self._link_state_inferred = np.zeros(self._num_links)

        while len(self.QB) != 0:
            # 从 E_C 中选择一条链路使得 gamma_k^(1) 取得最大值，并将此链路认为是拥塞链路
            # temp_state = [-1 for _ in range(self.link_num)]
            temp_state = [-1 for _ in range(self._num_links)]
            for i in self.__suspected_congested_links:
                # temp_state[i] = np.log(1/self.link_congestion_Pr[i])
                temp_state[i] = np.log(1/self._congestion_prob_links[i])
            link = temp_state.index(max(temp_state))
            self._link_state_inferred[link] = 1
            # self.congested_link_inferred.append(link + 1)
            self._congested_link_inferred.append(link + 1)

            self.__suspected_congested_links = self.__suspected_congested_links.difference(set([link]))

            domain = set()
            # for i in self.network.get_Path(link):
            for i in self._net.get_paths(link+1):
                if i not in self.__un_congested_paths:
                    domain.add(i)
            self.QB = self.QB.difference(domain)


    def get_paths(self):
        """
        拥塞路劲的集合和非拥塞路劲的集合
        :return:
        """
        # for index, item in enumerate(self.path_state):
        for index, item in enumerate(self._state_paths):
            self.__paths.add(index)
            if item == 1:
                self.__congested_paths.add(index)
            else:
                self.__un_congested_paths.add(index)


    def get_links(self):
        # 所有链路的集合
        # self.__links = set([i for i in range(self.link_num)])
        self.__links = set([i for i in range(self._num_links)])

        for i in self.__un_congested_paths:
            for j in self.__links:
                # if self.routing_matrix[i][j] == 1:
                if self._routing_matrix[i][j] == 1:
                    self.__un_congested_links.add(j)

        self.__suspected_congested_links = self.__links.difference(self.__un_congested_links)


