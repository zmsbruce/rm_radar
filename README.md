<div align="center">
<img src="./doc/images/readme/cover.jpeg">
</div>

<br>

<div align="center" float="left">
<a href="https://www.robomaster.com/zh-CN">
<img src="./doc/images/readme/RoboMaster-mecha-logo.png" width=25% />
</a>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="./doc/images/readme/hitcrt-vision-logo.png" width=25% />
</div>

**<div align="center">哈尔滨工业大学 竞技机器人队 视觉组</div>**

# <div align="center"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;雷达 📡</div>

_<div align="right">by zmsbruce</div>_

### <div align="center"> 1. 简介 📓 </div>

代码包含目标的识别、定位和跟踪，目前**不包括串口通信、相机 / LiDAR 驱动、自主决策、数据存取、UI**。

- **识别 🔎**：使用神经网络，根据相机传来的场地图像，对机器人和装甲板进行识别，得到机器人的矩形框和类别；

- **定位 🧭**：利用激光雷达提供的点云信息，结合机器人的图像位置信息、相机与雷达的相对位置，以及雷达坐标和场地坐标的相对位置，来确定机器人在场地中的位置；

- **跟踪 👁️**：对当前帧识别到的机器人进行位置滤波和跟踪，避免误识别和位置检测跳变带来的影响；

根据官方披露的数据，以该项目为核心构造的雷达站系统，其平均标记准确率为 **83.66%**，最高单场标记准确率 **93.55%**，局均易伤时间 **686.4** 秒，局均额外伤害 **177**.

有关雷达算法的详细介绍，可以查看 doc 路径下的文档。

### <div align="center"> 2. 性能 🚀 </div>

得益于使用 CUDA 进行前后处理，以及使用 TensorRT 进行模型的推理，我们保证了**较快的处理速度**，**在配置较低的运算平台上也有不俗的性能发挥**。具体而言，我们使用训练后的 yolov8m 作为机器人和装甲板的识别网络，使用 CUDA 版本为 12.3、TensorRT 版本为 8.6，对一次完整的机器人检测、跟踪与定位进行**计时 ⏱️**，得到数据如下：

- 在 NVIDIA GeForce GTX 1650, AMD Ryzen 7 4800H 上，平均时间为**32ms**；
- 在 NVIDIA GeForce RTX 3060Ti, 11th Gen Intel Core i5-11600KF 上，平均时间为**11ms**。

此外，我们对**目标定位**和**跟踪**均进行了**优化**，以提升**速度**和**准确性**。有关优化的具体实现，可以查看 doc 目录下文档。

### <div align="center"> 3. 安装 🔨 </div>

#### <div align="center"> 3.1 依赖 🔃 </div>

- 在 Linux 系统下编译和运行本项目；
- 代码基于 C++20 ，需要使用 GCC 11 或更高版本；
- 需要 CMake 版本不低于 3.19，以完整支持 CUDA Toolkit 包的获取与配置；
- 需要 CUDA 版本不低于 12.2，TensorRT 版本不低于 8.5；
- 需要 OpenCV, PCL, Eigen 库以支持图像、点云处理和目标跟踪；
- 需要 GTest 库以支持测试；
- 需要安装 TBB 库以支持 GCC 下的 `std::execution`；

#### <div align="center"> 3.2 编译与运行 🛠️ </div>

输入以下命令进行代码的**获取**、**编译**与**运行**：

```sh
# clone from website
git clone https://github.com/zmsbruce/rm_radar.git
cd rm_radar

# compile
mkdir build && cd build
cmake ..
make -j$(nproc)

# test
ctest

# run sample
../bin/sample
```

### <div align="center"> 4. 联系我 📧 </div>

如果对代码有疑问，或者想指出代码中的错误，可以提出 Issue 或通过邮件联系我：[zmsbruce@163.com](zmsbruce@163.com)。
