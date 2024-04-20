# 任务描述
一个图像分类系统，能够对输入图像进行类别预测。具体的说，利用数据库的2250张训练样本进行训练；对测试集中的2235张样本进行预测。
# 如何运行
1. 如果data目录中没有你所需要的处理好的数据集，需要把sign参数手动修改为True，你可以改动numOfBag的大小。
```
config=Namespace(
    image_dir="15-Scene",
    data_dir="data",
    numOfBag=300,
    C=10,
    gamma=0.001,
    kernel='rbf',
    categories=['Bedroom','Suburb','Industrial','Kitchen','Living room','Coast','Forest','Highway'
                ,'Inside city','Mountain','Open country','Street','Tall Building','Office','Store'],
    sign=False #判断是否需要处理数据集
)
```
2. 直接运行即可