```python
def __init__(self):
"""
        该网络类的所有属性
        self.tree_vector:list  树向量
        self.leaf_nodes:ndarray   所有的叶子节点数组
        self.num_paths:int  路径数量
        self.num_links:int  链路数量
        self.route_matrix:ndarray  路由矩阵
        self.depth:int  树的深度

        链路拥塞相关属性
        self.links_cong_pro:ndarray  链路拥塞概率数组
        self.loss_model = loss_model   丢包率模型
        self.threshold = threshold   丢包率门限

        self.links_state_true:ndarray    真实链路拥塞状态数组
        self.links_loss_rate:ndarray   链路丢包率数组
        self.links_cong_true:ndarray    真实拥塞链路编号数组

        路径拥塞相关属性
        self.paths_loss_rate:ndarray   路径丢包率数组
        self.paths_state_true:ndarray  真实路径拥塞状态数组
        self.paths_cong_true:ndarray  真实拥塞路径编号数组
        self.paths_no_cong_true:ndarray  真实非拥塞路径编号数组

        self.paths_state_obv:ndarray 观测路径拥塞状态数组
        self.paths_cong_obv:ndarray  观测拥塞路径编号数组
        self.paths_no_cong_obv:ndarray  观测非拥塞路径编号数组
        """
    #使用两个初始化方法对类进行初始化
    #（1）从json中读取数据 进行类初始化
    #（2）根据链路拥塞概率生成数据，进行类初始化
    pass

#使用从json文件中读取数据进行初始化的方法
def json_init(self,basic_info:dict,cong_data_info:dict,cong_config_info:dict):
    """
    从json文件中将文件导入
    :param basic_info:dict   topo相关信息
    :param cong_data_info:dict    一组拥塞数据相关信息
    :param cong_config_info:dict  拥塞数据配置相关信息
    :return:
    """
    

#数据自动生成初始化相关的方法
def auto_init(self,tree_vector:list,loss_model:str,threshold:float,links_cong_pro=None):
    """
    根据概率自动生成网络相关数据
    :param tree_vector: list 树向量
    :param loss_model: str 丢包率模型
    :param threshold:  float 丢包率门限
    :param links_cong_pro: list 链路拥塞概率
    :return:
    """    
```
# 目录

+ dataset_zoom   从topo-zoo中下载的所有gml文件

+ topoImgs  所有topo的树形图像

+ plot_net.py  画图代码

+ topoinfo.csv  易长胜的树属性统计

+ cong_net.py   生成拥塞网络类

+ gen_data.py 保存从topoinfo中的数据到all_DS中TOPO_INFO目录下

  + 保存all_topo.csv 保存 all_topo.xlsx 相同信息

  + all_topo.xlsx

    + ```
      #mesh_topo_info[(节点数量，链路数量，路径数，网络直径)]
      #tree_topo_info[(节点数量，链路数量，路径数,深度，广度)
      ```

  + all_topo.json  

    + treevector name filename

# cong_net的简单介绍
> 该类有两种初始化方法
>+ json_init 使用json初始化，从json中读取相关数据
>+ auto_init 使用自动初始化，根据参数随机生成数据
>

# Json的格式介绍
```python

all_data={
    "basic_info":{
        "tree_vector":[],#树向量
        "leaf_nodes":[], #叶子结点列表
        "num_paths":None,#路径数量
        "num_links":None,#链路数量
        "route_matrix":[[]],#路由矩阵
        "depth": None,#树的深度
    },
    "links_pro_scope":[[0, 0.1],[0, 0.2]],#拥塞概率范围
    "[0, 0.1]":{
        "cong_data_info": [{  #生成的拥塞信息,是一个列表，每一元素为每一次产生数据
            "links_state_true": [],#真实链路状态
            "links_loss_rate": [],#链路丢包率
            "links_cong_true": [],#真实拥塞链路数组
            "paths_loss_rate": [],#真实路径丢包率
            "paths_state_true": [],#真实路径状态
            "paths_cong_true": [],#真实拥塞路径列表
            "paths_no_cong_true": [],#真实非拥塞路径列表
            "paths_state_obv": [],#观测路径状态
            "paths_cong_obv": [],#观测拥塞路径列表
            "paths_no_cong_obv":[]#观测非拥塞路径列表
        },...],
        "cong_config":{
            "loss_model": "loss_model_1",#拥塞概率模型
            "threshold": 0.01,#拥塞门限
            "experiment": 10,#数据产生次数
            "links_cong_pro": [] #链路拥塞概率列表
        }
    },
    "[0, 0.2]":{
        "cong_data_info": [],
        "cong_config":{},
    }
    ...
}
```

# 使用demo
```python
def test_cong_net():
    tree_vector = [0, 1, 1, 3, 3]
    links_congest_pro = utils.gen_link_congest_pro_1([0, 0.2], len(tree_vector))
    net = Cong_net()
    net.auto_init(tree_vector, "loss_model_1", 0.015,links_congest_pro)
    print(net)
    pass
```