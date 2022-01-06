# -*- coding: utf-8 -*-
"""
Generate Tree Network

Routing Matrix, Link State, Path State
"""

import numpy as np
from numba import jit
from sys import exit

class network():
    def __init__(self):
        pass

    def plot_Tree(self): #画出树型拓扑
        import matplotlib.pyplot as plt
        plt.figure() # 新开一个作图窗口

        link = self.linkSetVec[:, 0]
        congestedLink = self.link_state

        x,y=self.treeLayout(link) #调用树布局函数
        n=len(link)
        i=0
        for f in link:
            if f==0:
                i+=1
        leaves=[]
        while i<n+1:
            b=0
            j=0
            while j<n:
                if i==link[j]: #如果有了第一个父节点为i的点就跳出循环
                    b=j
                    break
                j+=1
            if b==0: #如果没有找到父节点为i的节点，就为叶子节点
                leaves.append(i)
            i+=1

        num_layers=1/min(y)-1
        num_layers=int(num_layers+0.5)

        i=0
        chains=[]
        while i<len(leaves): #判断链路并加入到chains队列中
            index=leaves[i]-1
            chain=[]
            chain.append(index + 1)
            parent_index=link[index]-1
            j=1
            while parent_index != 0:
                chain.append(parent_index+1)
                parent_index=link[parent_index]-1
                j += 1
            chain.append(1)
            chain.reverse()
            chains.append(chain)
            i += 1

        y_new=y
        y_new=np.zeros(len(y))
        i = 0
        while i<len(y): #调整y的值
            r=0
            j=0
            b=0
            while j<len(leaves):
                r=0
                for c in chains[j]:
                    if c==i+1:
                        b=1
                        break
                    elif c!=i+1:
                        r += 1

                if b==1:
                    break
                elif b==0:
                    j += 1
            y_new[i]=0.9-(r-1)/(num_layers+1)
            i += 1
        plt.figure()
        #画线
        i=0
        while i<len(leaves):
            j=0
            while j+1<len(chains[i]):
                line_x=[]
                line_y=[]
                line_x.append(x[chains[i][j]-1])
                line_x.append(x[chains[i][j+1]-1])
                line_y.append(y_new[chains[i][j]-1])
                line_y.append(y_new[chains[i][j+1]-1])
                if j+2==len(chains[i]) and congestedLink[chains[i][j+1]-1]==0:
                    plt.plot(line_x,line_y,'g-',linewidth=0.5)
                elif j+2==len(chains[i]) and congestedLink[chains[i][j+1]-1]>0:
                    plt.plot(line_x,line_y,'r-',linewidth=0.5)
                elif congestedLink[chains[i][j+1]-1]==0:
                    plt.plot(line_x,line_y,'g-')
                elif congestedLink[chains[i][j+1]-1]>0:
                    plt.plot(line_x,line_y,'r-')
                j += 1
            i += 1
        #画点
        i=0
        while i<len(leaves):
            j=0
            while j<len(chains[i]):
                point_x=[]
                point_y=[]
                point_x.append(x[chains[i][j]-1])
                point_y.append(y_new[chains[i][j]-1])
                if j+1==len(chains[i]):#画出叶子节点
                    plt.plot(point_x,point_y,'bo',linewidth=0.5)
                else:#画出非叶子节点
                    plt.plot(point_x,point_y,'ko')
                j += 1
            i += 1

        for i in range(len(x)): #加上标号
            plt.text(x[i]*1.02, y_new[i]*1.02, str(i+1))

        plt.plot(x[-1],y_new[-1],'bo') # 'Destination Node'
        plt.plot(x[0],y_new[0],'ko') # 'Internal Node'
        plt.plot(x[0:1],y_new[0:1],'g-')
        plt.plot(label='line')
        xx=[x[0],x[0]]
        yy=[0.9-(-2)/(num_layers+1),y_new[0]]
        if congestedLink[0]==0:
            plt.plot(xx,yy,'g-')
        else:
            plt.plot(xx,yy,'r-')
        plt.plot(xx[0],yy[0],'k*') #'Root Node'
        plt.text(xx[0]*1.05,yy[0],str(0),family='serif',style='italic',ha='right',wrap=True)

        plt.legend()
        plt.xticks([])
        plt.yticks([])

        plt.show(block=False)
        input('按 <ENTER> 以继续运行后续程序代码')

    def treeLayout(self, parent): #生成树型拓扑中节点的坐标(x,y)
        #Lay out tree or forest
        #parent is the vector of parent pointers,with 0 for a root
        #post is a postorder permutation on the tree nodes
        #xx and yy are the vector of coordinates in the unit square at which
        #to lay out the nodes of the tree to make a nice picture
        #Optionally h is the height of the tree and s is the number of vertices
        #in the top-level separator
        pv=[]
        n=len(parent)
        parent,pv=self.fixparent(parent)

        #j=find(parent) in matlab
        j=np.nonzero(parent)
        jj=[x+1 for x in j[0]]

        #A=sparse(parent(j),j,1 n,n); in matlab
        A=np.zeros((n,n))
        for i in range(len(jj)):
            A[parent[i]-1,jj[i]-1]=1
        A=A+A.T+np.eye(n)
        post=self.etree(A)

        #Add a dummy root node and identify the leaves
        for _ in range(len(parent)):
            if parent[_]==0:
                parent[_]=n+1       #change all 0s to n+1s

        #in postorder computer height and descendant leaf intervals
        #space leaves evenly in x
        isaleaf = [1 for _ in range(n+1)]
        for i in range(len(parent)):
             isaleaf[parent[i]-1]=0

        xmin = [n for _ in range(len(parent)+1)]
        xmax = [0 for _ in range(len(parent)+1)]
        height=[0 for _ in range(len(parent)+1)]
        nkids =[0 for _ in range(len(parent)+1)]
        nleaves = 0

        for i in range(1,n+1):
            node = post[i-1]
            if isaleaf[node-1]:
                nleaves = nleaves + 1
                xmin[node-1] = nleaves
                xmax[node-1] = nleaves

            dad = parent[node-1]
            height[dad-1] = max(height[dad-1],height[node-1]+1)
            xmin[dad-1] = min(xmin[dad-1],xmin[node-1])
            xmax[dad-1] = max(xmax[dad-1],xmax[node-1])
            nkids[dad-1] = nkids[dad-1] + 1

        #compute coordinates leaving a little space on all sides
        treeht = height[n] - 1
        deltax = 1/(nleaves+1)
        deltay = 1/(treeht+2)
        x=[]
        y=[]

        #Omit the dummy node
        for _ in range(len(xmin)):
            x.append(deltax*(xmin[_]+xmax[_])/2)
        for _ in range(len(height)):
            y.append(deltay*(height[_]+1))

        for i in range(-1,-1*len(nkids)):
            if nkids[i]!=1:
                break
        xx=[]
        yy=[]
        flagx=1
        flagy=1

        for _ in  pv:
           for i in range(len(pv)):
               if pv[i]==flagx:
                   xx.append(x[i])
                   flagx=flagx+1
           for i in range(len(pv)):
               if pv[i]==flagy:
                   yy.append(y[i])
                   flagy=flagy+1


        return xx,yy


    def etree(self, mat):
        ##为其上三角形是 A 的上三角形的对称方阵返回消去树？？？
        if mat.shape[0]==mat.shape[1]:
            return [x for x in range(1,mat.shape[0]+1)]
        else:
            pass

    def fixparent(self, parent):
        #Fix order of parent vector
        #[a,pv]= fixparent(B) takes a vector of parent nodes for an elimination
        #tree, and re-orders it to produce an equivalent vector
        #a in which parent nodes are always higher-number than child nodes
        #if B is an elimination tree produced by the tree funtion, this step will not
        #be necessary. PV is a permutation vector,so that A=B(pv)
        n=len(parent)
        a=parent
        a[a==0] = n+1
        pv = [x for x in range(1,n+1) ]
        niter = 0
        while(True):
            temp=[_ for _ in range(1,n+1)]
            x=[]
            for i in range(len(a)):
                if (a[i]<temp[i]):
                    x.append(i+1)
                else:
                    x.append(0)

            k=np.nonzero(x)
            kk=[xx+1 for xx in k]
            if len(kk[0])==0:
                break
            kk=kk[0][0]
            j=a[kk-1]

            a=a.tolist()
            tem=a[kk-1]
            del(a[kk-1])
            a.insert(j-1,tem)
            a=np.array(a)

            tem=pv[kk-1]
            del(pv[kk-1])
            pv.insert(j-1,tem)

            te = [0 for _ in range(len(a))]
            for _ in range(len(a)):
                if j<=a[_]<kk:
                    te[_]=1
            for _ in range(len(a)):
                if a[_]==kk:
                    a[_]=j
            for _ in range(len(te)):
                if te[_]==1:
                    a[_]=a[_]+1
            niter = niter + 1


        a[a>n] = 0
        return a,pv

    def gen_Tree(self, outDegree,pathNum): #产生随机树型拓扑
        ###  outDegree: 最大的出度
        ###  pathNum:   端到端的路径数量，等于叶子节点的数量

        ###  VTRee:     树型拓扑的向量化表示
        ###  depth:     树型拓扑的深度

        ### 生成随机树的思维为“一层一层地增加节点，直到当前地叶子节点数目和给定的端到端路径数目相等”

        # if outDegree > pathNum:
        #     print("Warning: 出度小于等于路径数 \n")

        while True:
            linkSet = np.mat([0, 1])    #默认至少包含一条root 一根链路

            nodeNum = 1         #当前叶子节点的个数
            currentNodeSet = np.mat([1])  #当前节点的集合

            flag = 0            #判断当前是否是内部节点，默认不是
            pn = 0              #当前已生成树的路劲数目（=叶子节点的个数）

            distNodeSet = []    #目标节点集合
            while pn<pathNum:
                nextNodeSet = np.mat([])

                tempPN = pn
                addedNode = 0
                odegree = 0

                for i in range(1, self.length(currentNodeSet)+1):         #树生长，一个节点一个节点的循环
                    if currentNodeSet.size == 1:
                        odegree = 2
                    else:
                        re = 0 # 解决下面的循环存在异常锁死的问题
                        while True:         #生成合法的子节点
                            re += 1
                            if  re >= pow(pathNum,2):
                                pn = 2*pathNum
                                i = self.length(currentNodeSet)+2
                                break # 如果锁死，则控制进行下一轮拓扑生成工作

                            if pathNum - tempPN < 3:
                                odegree = np.random.randint(1, 4)       #随机生成从1到3的正整数
                            else:
                                odegree = np.random.randint(1, outDegree+1)         #随机生成从1到outDegree的正整数

                            if tempPN + odegree + self.length(currentNodeSet) - i <= pathNum:
                                break       #当前生成路径数目不超过给定路径数目，合法

                    tempPN = tempPN + odegree

                    if pn < pathNum and odegree > 1:         #当前节点是内部节点
                        flag = 1
                        childNodeSet = np.mat([k for k in range(nodeNum+1+addedNode, nodeNum+addedNode+odegree+1)])
                        if nextNodeSet.shape == (1,0):
                            nextNodeSet = childNodeSet.T
                        else:
                            nextNodeSet = np.vstack((nextNodeSet,childNodeSet.T))
                        linkSet =np.vstack((linkSet , np.hstack((np.tile(currentNodeSet[i-1], (odegree, 1)), childNodeSet.T))))
                        addedNode = addedNode + odegree
                    else:       #当前节点是叶子节点
                        if pn < pathNum:
                            distNodeSet += list(currentNodeSet.tolist()[i-1])
                        pn = pn + 1

                if pn < pathNum and not flag:
                    distNodeSet += list(currentNodeSet.tolist()[i-1])
                    continue
                else:           #如果当前节点是内部节点
                    flag = 0
                    nodeNum = nodeNum + addedNode
                    currentNodeSet = nextNodeSet

                if pathNum - currentNodeSet.size == pn:
                    for node in list(currentNodeSet.tolist()):
                        distNodeSet += node # 如果已经满足路径数目要求，则当前所有节点均为目的节点
                    pn = pathNum

            if pn == pathNum and len(set(distNodeSet)) == pathNum: # 判断是否生成了满足要求的随机树
                distNodeSet.sort()
                self.linkSetVec = linkSet.getA()

                self.dist = distNodeSet
                self.path_num, self.link_num = pathNum, np.size(self.linkSetVec, axis=0)

                self.get_RM()
                break

    def length(self, mat): # 同matlab中的length函数，返回矩阵维度的最大值
        return mat.shape[0] if mat.shape[0] >= mat.shape[1] else mat.shape[1]

    @jit
    def get_RM(self): # 生成路由矩阵
        routing_matrix = np.zeros((self.path_num, self.link_num))

        for pn in range(self.path_num):
            link = self.dist[pn] - 1 # 根据节点编号规则来，根节点从0开始编号，其有唯一一个子节点1
            routing_matrix[pn][link] = 1

            link = self.linkSetVec[link][0] - 1
            while link != -1:
                routing_matrix[pn][link] = 1
                link= self.linkSetVec[link][0] - 1

        self.routing_matrix = routing_matrix


    def gen_State(self, stype, max_state): # 给链路与路径赋拥塞状态
        self.stype, self.max_state = stype, max_state

        if stype == 0: # 布尔状态 {0,1}
            self.link_state = 1*(np.random.rand(self.link_num) <= \
                                 self.link_congestion_Pr)
        elif stype == 1: # 整数状态 {0,1,2,...}
            self.link_state = np.multiply(np.random.randint(max_state, size=(self.link_num))+1, \
                                   1*(np.random.rand(self.link_num) <= self.link_congestion_Pr))
        else:
            print('未知链路状态类型...')
            exit(2)

        self.congested_link, = np.where(self.link_state > 0) # 真实的拥塞链路集合
        self.congested_link += 1

        self.path_state = np.dot(self.routing_matrix, self.link_state)
        if stype == 0: # 布尔状态 {0,1}
            self.path_state[np.where(self.path_state > 1)] = 1

    def get_Path(self, l): # 获取经过链路l的所有路径
        path, = np.where(self.routing_matrix[:, l] > 0)
        return path.tolist()

    def get_Child(self, k): # 获取节点k的所有子节点
        index, = np.where(self.linkSetVec[:, 0] == k)
        index = index

        child = list(self.linkSetVec[index, 1])
        return child

    def conf_Link_Congestion_Pr(self, congestion_Pr, ctype): # 给链路赋先验拥塞概率
        self.ctype = ctype

        if ctype == 0: # 全部链路具有相同的先验拥塞概率
            self.link_congestion_Pr = np.ones(self.link_num) * congestion_Pr
        elif ctype == 1: # 所有链路的拥塞链路服从均匀分布
            self.link_congestion_Pr = np.random.rand(self.link_num) * congestion_Pr
        elif ctype == 2: # 所有链路的先验拥塞概率总体上服从指数分布，均值
            self.link_congestion_Pr = np.random.exponential(congestion_Pr, self.link_num)
#        elif ctype == 3: # 所有链路的先验拥塞概率总体上服从高斯分布, 均值，标准方差
#            self.link_congestion_Pr = np.random.normal(congestion_Pr[0], congestion_Pr[1], self.link_num)
#            self.link_congestion_Pr = np.absolute(self.link_congestion_Pr) # 将负概率转成正概率
#        elif ctype == 4: # 所有链路的先验拥塞概率总体上服从Zipf分布
#            self.link_congestion_Pr = np.random.zipf(congestion_Pr, self.link_num)
        else:
            print('未定义链路先验拥塞概率场景...')
            exit(1)

    def show_para(self):
        print('\n-=-=-=-=-=-=-=-=-=-=-PAR.-=-=-=-=-=-=-=-=-=-=-')
        print('节点向量:', self.linkSetVec[:,0])
        print('路由矩阵:', [i for i in self.routing_matrix])
        print('\n目标节点:', [i for i in self.dist])
        print('路径状态:', self.path_state, '\n')
        for i in set(range(0, self.link_num+1)).difference(\
                           set(self.dist)):
            print('节点 %d'%(i), '的子节点', [j for j in self.get_Child(i)])

        print('')
        for i in range(self.link_num):
            path_index = self.get_Path(i)
            path_index = np.array(path_index, dtype='int')
            path_list = [self.dist[p] for p in path_index]

            print('经过链路 l_%d'%(i+1), '的路径为 P_', [p for p in path_list])

        print('\n真实的拥塞链路先验拥塞概率:', [ int(pr *10000)/10000.0 for pr in self.link_congestion_Pr])
        print('真实的链路拥塞状态:', [ ls for ls in self.link_state])
        print('真实的拥塞链路:', [i for i in self.congested_link])
        print('-=-=-=-=-=-=-=-=-=-=-END-=-=-=-=-=-=-=-=-=-=-')

if __name__ == '__main__':

    sim = network()

    outDegree, pathNum = 4, 20
    sim.gen_Tree(outDegree, pathNum)

    congestion_Pr, ctype = 0.3, 0
    sim.conf_Link_Congestion_Pr(congestion_Pr, ctype)

    sim.gen_State(1, 1)

    sim.show_para()

    sim.plot_Tree()
