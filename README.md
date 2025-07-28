智眼护航方案深度融合激光雷达、AI 视觉与边缘智能,打造“边缘智能+雷视融合”架构，突破航道 180°监测、3D定位与轨迹预测技术，实现对各级航道覆盖监测，高精度、准实时响应的船舶感知预警体系。
构建感知网与数据平台，推动航道安全向智慧管理及信息服务平台转型，服务智慧航运与海洋强国战略。

# 项目依赖安装指南 (Windows + Visual Studio 2022)

## 前置要求
- Visual Studio 2022 (安装时勾选 "C++ 桌面开发" 和 "Windows SDK")
- [CMake 3.20+](https://cmake.org/download/)
- [Git for Windows](https://git-scm.com/download/win)

---

## 1. 安装 OpenCV 4.8.0
```powershell
# 下载预编译库 (推荐)
choco install opencv -y --version=4.8.0

# 手动编译 (可选)
git clone -b 4.8.0 https://github.com/opencv/opencv.git
cmake -B build -DCMAKE_INSTALL_PREFIX="C:/opencv" -DBUILD_LIST=core,highgui,imgproc
cmake --build build --target INSTALL --config Release

# 使用 vcpkg (推荐)
vcpkg install vtk[qt] --triplet=x64-windows

# 手动编译
git clone -b v9.3.0 https://github.com/Kitware/VTK.git
cmake -B build -DCMAKE_INSTALL_PREFIX="C:/vtk" -DBUILD_SHARED_LIBS=ON
cmake --build build --config Release --target INSTALL

# 使用 vcpkg (推荐)
vcpkg install pcl[core,visualization] --triplet=x64-windows

# 手动编译
git clone -b pcl-1.15.1 https://github.com/PointCloudLibrary/pcl.git
cmake -B build -DCMAKE_INSTALL_PREFIX="C:/pcl" -DBUILD_visualization=ON -DWITH_VTK=ON
cmake --build build --config Release --target INSTALL

