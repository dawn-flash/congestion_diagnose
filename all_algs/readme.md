# 算法基本介绍
基类：Base_Alg
+ 子类1：SCFS
+ 子类2：CLINK

# 基本使用
+ 构建好Cong_net网络类后，直接初始化算法类
    + 调用diagnose() 进行诊断
    + 调用evluation() 进行算法分析 
```python
def __init__(self, net:Cong_net):
    """
    所有算法的基类，
    diagnose()  该函数进行拥塞链路的诊断
    evaluation() 该函数用于诊断结果的相关指标计算

    算法生成的推测信息：
    self.links_cong_inferred：list  # 诊断为拥塞的链路编号列表
    self.links_state_inferred：ndarray  # 诊断的链路拥塞的状态
    self.dr:float
    self.fpr:float
    self.f1_score:float
    :param net:  拥塞网络类
    """

def diagnose(self):
    """
    拥塞链路诊断，由子类继承
    :return:
    """
    pass

def evaluation(self):
    """
    拥塞链路评估方法1
    dr和fpr的计算方式1
    dr=(推测拥塞 & 真实拥塞)/(真实拥塞数量)
    fpr=(推测拥塞-真实拥塞)/(真实不拥塞数量)
    :return:
    """
   
```

# CLink demo
```python
def test_clink1():
    random.seed(0)
    np.random.seed(0)
    tree_vector = [0, 1, 1, 3, 3]
    links_congest_pro = utils.gen_link_congest_pro_1([0, 0.5], len(tree_vector))
    net = Cong_net()
    net.auto_init(tree_vector, "loss_model_1", 0.015, links_congest_pro)
    print(net)

    clink=CLINK(net)
    clink.diagnose()
    clink.evaluation()
    print(clink)
```

# SCFS demo
```python
def test_scfs():
    """
    scfs算法的测试例子
    :return:
    """
    # random.seed(2)
    # np.random.seed(2)
    tree_vector = [0, 1, 1, 3, 3]
    links_congest_pro = utils.gen_link_congest_pro_1([0, 0.5], len(tree_vector))
    net = Cong_net()
    net.auto_init(tree_vector, "loss_model_1", 0.015, links_congest_pro)
    print(net)

    scfs=SCFS(net)
    scfs.diagnose()
    scfs.evaluation()
    print(scfs)
```
