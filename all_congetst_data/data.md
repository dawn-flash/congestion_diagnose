
# 目录结构
+ data.md 说明文档
+ all_topo.json 简单的网络拓扑json数据（简单数据）
+ operate_congest_data 操作生成测量数据的文件
+ data 保存所有的测量数据到该文件夹
+ all_detail_topo.json  保存所有的topo属性信息

# all_basic_topo.json 文件

保存所有的topo属性信息

```
{
	"topo_name":{
		"name":str    名字
		"graph_name":str   图片名
		"tree_vector":list(n,):
		"route_matrix":list(path,link):
		"node_num":
		"path_num":
		"width":
		"depth":
		"path_list": 路径列表   
		"leaf_list": list(n)  叶子结点列表
	}
}
```

注意：

+ route_matrix维度(path,link) path排序从上到下，从左到右

# data文件

+ 真实的链路拥塞矩阵和路径拥塞矩阵

json文件格式

```
congest_data = {
        "tree_name": tree_name,
        "tree_vector": tree_vector,
        "route_matrix": route_matrix,  路由矩阵
        "measure_time": measure_time,  测量次数
        "K":20,
        "link_prob_isomorphism":list(n,), 同构的概率列表
        "link_prob_isomerism":list(n,2),  异构的概率列表
        
        "isomorphism": {
        	"index":{ 每个index为一组概率数据
        	"link_prob_list": list(link,) 链路拥塞概率
        	"link_measure_data": list(link,time)
        	"path_measure_data":  list(path,time)
        	}
        },
        "isomerism": {
        	"index":{每个index为一组概率
        		"prob_scope":list(2,) 概率取值范围
        		"link_prob_list": list(K,link) 链路概率拥塞组
        		"link_measure_data": list(K,link,time)
        		"path_measure_data": list(K,path,time)
        	}
        }
    }
  	其中k=20  未保存
```

关键信息：

+ ```
  同构概率：link_prob_isomorphism = [0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]
  
  异构概率link_prob_isomerism = [[i * 0.5, i * 1.5] for i in link_prob_isomorphism]
  ```

+ 同构的链路失效情况：一共12组概率，每组概率生成一个链路失效矩阵link_measure_data（link,time）time=1000  同时得到路径的失效情况。
+ 异构的链路失效情况：一共12组概率范围，每组概率范围取K=20次随机情况，生成link_prob_list组 。每一个概率范围的20组数据生成链路失效矩阵：link_measure_data（K,link,time） 和路径失效矩阵path_measure_data(K,path,time)。

# obv文件

观测的路径拥塞的文件

## 生成 路径观测矩阵

- 无观测误差，就是真实的路径拥塞矩阵（path，time=1000）
- 有观测误差（暂时所有路径观测误差一样，范围 从 0.05 到 0.25，以0.025为一个间隔）
  + [0.0，0.001, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]  除第一个以外，都以0.025为一个间隔。

```
obv_data = {
        "tree_name": tree_name,
        "tree_vector": tree_vector,
        "route_matrix": route_matrix,  路由矩阵
        "measure_time": measure_time,  测量次数
        "link_prob_isomorphism":list(n,), 同构的概率列表
        "link_prob_isomerism":list(n,2),  异构的概率列表
        "K"：20
        
        "error_list": list(n,) 观测误差概率列表
        "isomorphism": {
        	"index":{每一组链路拥塞概率数据
                "link_prob_list": list(link,)
                obv_error_data:{
                	path_obv_data:"path_measure_data":  list(path,time)
                	...
                }     
        	}
        },
        "isomerism": {
        	"index":{每一组的链路拥塞范围
        		"prob_scope":list(2,) 概率取值范围
        		"link_prob_list": list(K,link) 概率组
				obv_error_data:{
					path_obv_data:"path_measure_data": list(K,path,time)
					...
				}
        	}
        	...
        }
    }
```

