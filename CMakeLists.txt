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

