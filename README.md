# Tree(3) Problem 演示说明

这份代码不是在直接求数学上那个真正的 `TREE(3)` 数值。真正的 `TREE(3)` 大到远远超过日常数学、物理和计算机里的常见大数，不能靠普通程序“算出来”。

本项目做的是一个适合演示的简化模型：用 3 种颜色的有根树来展示 Tree(3) 问题里最重要的几个直觉，包括树的结构爆炸、嵌入关系、CPU 枚举为什么很快变慢，以及 CUDA 为什么能把某些计数任务加速很多。

## 基本概念

一棵树可以理解成一个没有环的层级结构。这里使用的是“有根树”：每棵树都有一个根节点，下面可以接若干子树。

Tree(3) 中的 `3` 可以直观理解为节点有 3 种颜色。本代码里用 `A`、`B`、`C` 表示三种颜色：

```text
A = red
B = green
C = blue
```

代码中的树用 Python 元组表示：

```python
(1, ((0, ()), (2, ())))
```

意思是：根节点颜色是 `B`，它有两个叶子节点，颜色分别是 `A` 和 `C`。

## 本代码演示了什么

`plot.py` 主要包含三类功能：

1. 随机生成 3 色有根树，并用 Matplotlib + NetworkX 画出来。
2. 判断一棵树是否能嵌入到另一棵树里，并画出 embedding matrix。
3. 计算 `node <= SUM_NODE` 的 3 色有根树总数，并用 CPU 和 GPU 做速度对比。

画图窗口支持交互：

```text
鼠标滚轮：缩放
左键拖动：平移
左键单击：放大
右键单击：缩小
中键或 R：重置视图
```

## CLI 用法

在 `Tree_plot` 目录的上一级运行：

```powershell
python Tree_plot\plot.py
```

查看所有参数：

```powershell
python Tree_plot\plot.py --help
```

只做 Tree(3) 计数演示，不打开画图窗口：

```powershell
python Tree_plot\plot.py --sum-node 7 --count-only
```

演示大节点数 GPU 计数：

```powershell
python Tree_plot\plot.py --sum-node 200 --count-only
```

生成最多 200 个节点的随机树，并打开可缩放、可拖动的图：

```powershell
python Tree_plot\plot.py --max-nodes 200 --limit 6 --sum-node 200
```

跳过 embedding matrix，只看随机树：

```powershell
python Tree_plot\plot.py --max-nodes 200 --limit 6 --no-matrix
```

不保存 PNG，只显示窗口：

```powershell
python Tree_plot\plot.py --max-nodes 120 --no-save
```

常用参数说明：

```text
--max-nodes          随机生成树时允许的最大节点数
--sum-node           计算 node <= SUM_NODE 的树总数
--limit              随机画几棵树
--seed               随机种子，方便复现实验
--colors             节点颜色数量；Tree(3) 演示默认是 3
--cols               随机树图里的列数
--matrix-limit       embedding matrix 最多使用多少棵树
--count-only         只做计数和测速，不画图
--skip-count         跳过计数和测速
--no-plot            不打开随机树图
--no-matrix          不画 embedding matrix
--force-gpu-sampling 即使 max-nodes 较小，也强制使用 GPU 采样
```

## CPU 和 GPU 对比

当 `SUM_NODE <= 10` 时，代码会同时运行 CPU 和 GPU：

```powershell
python Tree_plot\plot.py --sum-node 8 --count-only
```

CPU 路径会真的把所有树都枚举出来，所以它给的是很直观的精确基准。GPU 路径使用动态规划计数，然后比较两边结果是否一致。

当 `SUM_NODE > 10` 时，代码只运行 GPU：

```powershell
python Tree_plot\plot.py --sum-node 50 --count-only
python Tree_plot\plot.py --sum-node 200 --count-only
```

原因是 CPU 枚举不是“慢一点”，而是会遇到组合爆炸。节点数增长时，可能的子树组合数量会急剧膨胀。CPU 如果坚持逐棵树构造、排序、去重，运行时间和内存都会很快失控。对课堂演示来说，这正是 Tree(3) 类问题最震撼的地方：规则看起来很简单，规模却爆炸得非常快。

## 为什么 GPU 能快很多

CPU 枚举的思路是“把每棵树都造出来”。这适合小规模，因为学生能看到真实对象；但大规模时，它会被大量重复结构和组合数淹没。

GPU 计数的思路不同。它不再逐棵构造树，而是只记录“某个节点数有多少种树”。如果大小为 `s` 的子树有 `a_s` 种，那么一个根节点下面的孩子可以看成这些子树类型组成的无序多重集合。代码用生成函数和动态规划把这个问题变成一批张量运算。

GPU 快的原因主要有三个：

1. 它避免了构造每一棵具体的树。
2. 它把很多相似的小计算合并成批量张量计算。
3. CUDA 有大量并行计算单元，适合处理这种重复、规则、数组化的工作。

所以在 `SUM_NODE=200` 这类演示里，GPU 可以几秒钟给出结果量级；如果 CPU 仍然走逐棵枚举，可能会跑很久，甚至几天也不现实。具体耗时取决于显卡、CUDA/PyTorch 版本和参数设置。

## 一个重要提醒

大节点数时，树的数量会大到超过普通浮点数能直接表示的范围。代码会在必要时用科学计数法或 `10^x` 的形式输出数量级。这个输出适合用来展示增长速度和 GPU 加速效果；如果要做严格数学证明或精确大整数计数，需要使用更专门的符号计算或大整数算法。

