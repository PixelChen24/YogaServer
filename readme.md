# 姿态评分APP后台服务器
**服务器调用功能还没弄**
## 文件结构
```shell
.
├── main.py
├── readme.md
├── requirements.txt
├── StdPoseDatabase
│   ├── Images
│   ├── Management.txt
│   └── Points
├── test_main.py
└── utils
    ├── CalculateAngle.py
    ├── DataBase.py
    └── Score.py
```
* 主程序运行在`main.py`
* utils存放函数
    * `CalculateAngle.py`计算关节角度。
    * `DataBase.py`用于管理数据。主要是当数据库中增加或者删除标准数据时，自动生成或删除姿势点数据。**当前未实现**。
    * `score.py`为动作打分。只实现了最朴素的打分方法。
* StdPoseDatabase存放数据文件
    * Images存放标准姿势图片
    * Points存放标准姿势点数据

详细函数实现见代码注释(如果有写注释的话)

---
## 角度定义
![角度定义](https://markdown-bed-pixel.oss-cn-shanghai.aliyuncs.com/public/971124BD-3B4F-4DCE-A3CF-986346040C79.jpeg)
---
