#define NOMINMAX
#define _CRT_SECURE_NO_WARNINGS
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <vector>
#include <atomic>
#include <fstream>
#include <chrono>
#include <string>
#include <iostream>
#include <deque> //以便随机访问和高效删除任意位置的元素。

// 假设SimplePoint和LidarFrame已在lidar_core.h中定义
#include "D:/Livox-SDK/sdk_core/src/lidar_core.h"
#include <filesystem>
#include <cmath>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <future>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/point_cloud.h>  
#include <memory> // 使用标准库的shared_ptr
#include <pcl/filters/voxel_grid.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <algorithm>

// 需修改
constexpr double LIDAR_FOV_DEG = 16.0;//雷达的视场角
constexpr double TOTAL_ROTATE_DEG = 150.0; //一圈总角度
constexpr double SCAN_TIME_SEC = 60; // 一圈总时间（同步与电机一圈的速度）
constexpr double SCAN_ORDER = 1.0; // 逆时针为1，顺时针为-1
constexpr double SCAN_NUM = 60; // 存储数量（一圈储存的数量）
constexpr double ROTATE_DEG_SEC = SCAN_ORDER * TOTAL_ROTATE_DEG / SCAN_TIME_SEC;//每秒旋转角度 = (扫描方向) × (一圈总旋转角度) / (扫描总时间)

// 电机安装的位置在激光雷达的后方8.85cm，在其下面的4.35厘米，在其右边的7.6cm
//三维向量 `SCAN_OFFSET`，用于表示激光雷达相对于电机中心的安装位置偏移
const Eigen::Vector3f SCAN_OFFSET(-0.0885f, 0.0435f, -0.076f);


size_t lidar_queue_max = 100; // 可根据实际速率初始化，激光雷达数据队列的最大容量
size_t image_queue_max = 100; //相机数据队列的最大容量


// // 1. 声明全局指针用于回调
//g_lidar_queue 	std::deque<LidarFrame>*	指向激光雷达数据队列的指针
//g_lidar_mtx	    std::mutex* 指向保护队列的互斥锁指针（？？？？？？？？？？？？）
//static	存储类说明符	限制作用域为当前文件，避免全局命名冲突
//nullptr	初始值	确保指针初始化为空，防止未定义行为
static std::deque<LidarFrame>* g_lidar_queue = nullptr;
static std::mutex* g_lidar_mtx = nullptr;

// get_time_ms 实现跨平台时间，绝对时间戳
// std::chrono::system_clock::now()	获取当前系统时间点
//.time_since_epoch()	计算从纪元(1970-01-01 00:00:00 UTC)到当前的时间间隔
static uint64_t get_time_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
}

// 辅助函数：将时间戳转为"YYYYMMDD_HHMMSS"字符串
//将毫秒级时间戳格式化为易读日期时间字符串的功能
std::string format_datetime(uint64_t timestamp_ms) {
    //毫秒转秒
    std::time_t t = timestamp_ms / 1000;
    std::tm tm_time;

#ifdef _WIN32
    localtime_s(&tm_time, &t);
#else
    localtime_r(&t, &tm_time);
#endif
    //格式化字符串
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm_time);

    //返回字符串
    return std::string(buf);
}

// 定义点云数据结构
//cv::Mat 是 OpenCV 库中的一个类，用于存储图像数据
//uint64_t timestamp 用于记录图像帧的捕获时间，单位为毫秒
struct ImageFrame {
    cv::Mat image;
    uint64_t timestamp; // ms
};

// 定义ColoredPoint结构体（SimplePoint + PointXYZRGB）,结构体用于存储点云数据和颜色信息
// 为了兼容PCL和cloudcompare的点云格式，使用了XYZRGB+反射率字段
struct ColoredPoint {
    float x, y, z;   //  三维空间坐标
    uint8_t r, g, b;  // RGB颜色分量
    uint8_t reflectivity; // 反射率
};

struct FusedCloud {
    //存储三维点云数据
    std::vector<ColoredPoint> points;
    //用于记录数据采集时刻
    uint64_t timestamp; // 推荐用图像帧的timestamp
};

// ColoredPoint转PCL点云（自定义到标准格式）
pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_points_to_pcl(const std::vector<ColoredPoint>& points) {
    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
    for (const auto& pt : points) {
        pcl::PointXYZRGB p;
        p.x = pt.x; p.y = pt.y; p.z = pt.z;
        p.r = pt.r; p.g = pt.g; p.b = pt.b;
        // 关键：设置p.rgb字段
        //PCL的RGB 值（即 R 占高 8 位，G 占中间 8 位，B 占低 8 位）
        uint32_t rgb = (pt.r << 16) | (pt.g << 8) | pt.b;
        p.rgb = *reinterpret_cast<float*>(&rgb);
        cloud->points.push_back(p);
    }

    cloud->width = cloud->points.size();
    cloud->height = 1;
    //is_dense = true，需确保所有点有效
    cloud->is_dense = false;
    return cloud;
}

// PCL点云转ColoredPoint（标准格式到自定义结构）
std::vector<ColoredPoint> pcl_to_colored_points(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud) {
    std::vector<ColoredPoint> points;
    for (const auto& p : cloud->points) {
        points.push_back({ p.x, p.y, p.z, p.r, p.g, p.b, 0 });
    }
    return points;
}

// 全局相机参数（建议放在文件顶部）
//旋转向量（3×1），表示相机绕旋转轴的旋转角度
cv::Mat rvecsMat = (cv::Mat_<double>(3, 1) << 1.17817311, -1.23591078, 1.2170962); 
//平移向量（3×1），表示相机在世界坐标系中的位置
cv::Mat tvecsMat = (cv::Mat_<double>(3, 1) << 0.13189467, -0.04692108, -0.17265252);
//内参矩阵（3×3），包含焦距、主点等参数
cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<
    1801.971429070619, 0.0, 999.2835130372968,
    0.0, 1802.914852279953, 559.837389417315,
    0.0, 0.0, 1.0);
//畸变系数（1×5），用于校正镜头畸变
cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) <<
    -0.24700116390837773, -0.9700558563715386, 0.0006355807164093516, 0.00018147578886886173, 5.346907990928945);

// 使用相机标定参数的点云投影函数
// 将 3D 空间中的 SimplePoint 转换为图像平面中的像素坐标 (u, v)
bool point_to_pixel(const SimplePoint& pt, int& u, int& v, int width, int height) {
    // 只打印一次点云原始数据，静态变量控制输出
    static bool printed = false;
    if (!printed) {
        std::cout << "[point_to_pixel] 点云原始数据: x=" << pt.x
            << ", y=" << pt.y
            << ", z=" << pt.z
            << ", reflectivity=" << static_cast<int>(pt.reflectivity)
            << std::endl;
        printed = true;
    }
    // 构造点的世界坐标，通过 emplace_back（接收构造函数参数并在容器内存中直接创建元素）直接在向量中构造 cv::Point3d 实例
    std::vector<cv::Point3d> objectPoints;
    objectPoints.emplace_back(pt.x, pt.y, pt.z);

    // 投影到像素坐标
    std::vector<cv::Point2d> imagePoints;
    cv::projectPoints(objectPoints, rvecsMat, tvecsMat, cameraMatrix, distCoeffs, imagePoints);

    u = static_cast<int>(imagePoints[0].x);
    v = static_cast<int>(imagePoints[0].y);

    return u >= 0 && u < width && v >= 0 && v < height;
}

// PCD存储融合存储，方便cloudcompare查看彩色图像
//将自定义的 ColoredPoint 结构体数据保存为 PCD 文件，并支持颜色信息（RGB）的存储
void save_pcd_rgb(const std::string& filename, const std::vector<ColoredPoint>& points) {
    std::ofstream ofs(filename);
    ofs << "VERSION .7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n";
    ofs << "WIDTH " << points.size() << "\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n";
    ofs << "POINTS " << points.size() << "\nDATA ascii\n";
    for (const auto& pt : points) {
        uint32_t rgb = (pt.r << 16) | (pt.g << 8) | pt.b;
        ofs << pt.x << " " << pt.y << " " << pt.z << " " << *reinterpret_cast<float*>(&rgb) << "\n";
    }
    ofs.close();
}
//将 ColoredPoint 结构体数据保存为 PCD 文件
void save_pcd_rgb_atomic(const std::string& filename, const std::vector<ColoredPoint>& points) {
    //采用 原子操作 确保文件写入的原子性，写入临时文件：将数据写入临时文件（.tmp）。
    std::string tmp_filename = filename + ".tmp";
    std::ofstream ofs(tmp_filename);
    ofs << "VERSION .7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n";
    ofs << "WIDTH " << points.size() << "\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n";
    ofs << "POINTS " << points.size() << "\nDATA ascii\n";
    for (const auto& pt : points) {
        uint32_t rgb = (pt.r << 16) | (pt.g << 8) | pt.b;
        ofs << pt.x << " " << pt.y << " " << pt.z << " " << *reinterpret_cast<float*>(&rgb) << "\n";
    }
    ofs.close();
    // 写完后原子重命名为正式文件（避免写入过程中因崩溃导致部分数据丢失）
    std::filesystem::rename(tmp_filename, filename);
}

//将自定义的 ColoredPoint 结构体数据保存为 PCD 文件（直接将数据写入最终文件，无临时文件缓冲），并包含额外的 reflectivity 字段
void save_pcd(const std::string& filename, const std::vector<ColoredPoint>& points) {
    std::ofstream ofs(filename);
    ofs << "VERSION .7\n";
    ofs << "FIELDS x y z r g b reflectivity\n";
    ofs << "SIZE 4 4 4 1 1 1 1\n";
    ofs << "TYPE F F F U U U U\n";
    ofs << "COUNT 1 1 1 1 1 1 1\n";
    ofs << "WIDTH " << points.size() << "\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n";
    ofs << "POINTS " << points.size() << "\nDATA ascii\n";
    for (const auto& pt : points) {
        ofs << pt.x << " " << pt.y << " " << pt.z << " "
            << static_cast<int>(pt.r) << " "
            << static_cast<int>(pt.g) << " "
            << static_cast<int>(pt.b) << " "
            << static_cast<int>(pt.reflectivity) << "\n";
    }
    ofs.close();
}
//从 PCD 文件 中读取 彩色点云数据，并解析 RGB 颜色信息
std::vector<ColoredPoint> load_pcd_rgb(const std::string& filename) {
    std::vector<ColoredPoint> all_points;
    //以二进制模式打开 PCD 文件，准备读取文件内容
    std::ifstream ifs(filename);
    std::string line;
    // 跳过头部
    while (std::getline(ifs, line)) {
        if (line == "DATA ascii") break;
    }
    //逐行读取文件内容，直到文件结束
    while (std::getline(ifs, line)) {
        std::istringstream iss(line);
        ColoredPoint pt;
        float rgbf;
        //从输入流中读取四组数据，分别赋值给 pt.x, pt.y, pt.z 和 rgbf
        iss >> pt.x >> pt.y >> pt.z >> rgbf;
        //将 float 类型的 rgbf 转换为 uint32_t 类型
        uint32_t rgb = *reinterpret_cast<uint32_t*>(&rgbf);
        pt.r = (rgb >> 16) & 0xFF;
        pt.g = (rgb >> 8) & 0xFF;
        pt.b = rgb & 0xFF;
        //pt.reflectivity = 0; // 没有反射率字段，默认0
        //将解析出的 pt 对象添加到 all_points 向量中
        all_points.push_back(pt);
    }
    //返回解析后的点云数据向量
    return all_points;
}

// 读取PCD文件，需要注意pcl-f/RGB和cloudcompare-U8/r g b格式不同
std::vector<ColoredPoint> load_pcd(const std::string& filename) {
    std::vector<ColoredPoint> all_points;
    std::ifstream ifs(filename);
    std::string line;
    // 跳过头部
    while (std::getline(ifs, line)) {
        if (line == "DATA ascii") break;
    }
    while (std::getline(ifs, line)) {
        std::istringstream iss(line);        
        //将每行数据解析为 x, y, z, r, g, b, reflectivity，并填充到 ColoredPoint 结构体中
        ColoredPoint pt;
        uint8_t r, g, b, reflectivity;
        //从输入流中读取七组数据，分别赋值给 pt.x, pt.y, pt.z, r, g, b, reflectivity
        iss >> pt.x >> pt.y >> pt.z >> r >> g >> b >> reflectivity;
        pt.r = static_cast<uint8_t>(r);
        pt.g = static_cast<uint8_t>(g);
        pt.b = static_cast<uint8_t>(b);
        pt.reflectivity = reflectivity; // 反射率
        all_points.push_back(pt);
    }
    return all_points;
}

// 1. 采集图像：只采集并入列，不做任何磁盘或耗时操作
//用于从 RTSP 流中读取视频帧并将其存储到队列中的线程函数
void rtsp_reader(const std::string& rtsp_url, std::deque<ImageFrame>& img_queue, std::mutex& img_mtx, std::atomic<bool>& running) {
    //打开指定的RTSP流
    cv::VideoCapture cap(rtsp_url);
    if (!cap.isOpened()) {
        std::cerr << "RTSP流打开失败\n";
        running = false;
        return;
    }
    //获取视频流的帧率（FPS）。若无法获取，设置默认值为 25
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    if (fps <= 0) fps = 25;
    uint64_t last_save_sec = 0;

    while (running) {
        cv::Mat frame;
        if (!cap.read(frame)) continue;

        // 用系统时间作为帧时间戳，使用 get_time_ms() 获取当前时间（毫秒），并转换为秒数
        uint64_t now = get_time_ms();
        uint64_t frame_sec = now / 1000;
        // 每秒只保存一帧
        if (last_save_sec == 0 || frame_sec != last_save_sec) {
            //通过比较当前帧的秒数与上一次保存的秒数，确保每秒仅保存一次
            std::lock_guard<std::mutex> lock(img_mtx);
            if (img_queue.size() < image_queue_max) {
                img_queue.push_back({ frame.clone(), now });
            }
            last_save_sec = frame_sec;
        }
    }
}

// 2. 激光雷达回调函数：将接收到的点云帧入列，在多线程环境下安全地将激光雷达帧（LidarFrame）添加到线程安全的队列中
void on_lidar_frame(const LidarFrame* frame, std::deque<LidarFrame>& lidar_queue, std::mutex& lidar_mtx) {
    std::lock_guard<std::mutex> lock(lidar_mtx);
    if (lidar_queue.size() < lidar_queue_max) {
        lidar_queue.push_back(*frame);
    }
}

//将激光雷达点云（LidarFrame）与图像（cv::Mat）融合，生成带有颜色信息的点云
std::vector<ColoredPoint> fuse_cloud_with_image(const LidarFrame& pc, const cv::Mat& image) {
    int width = image.cols;
    int height = image.rows;

    // 用于记录每个像素最近点的索引和深度
    std::vector<std::vector<std::pair<float, int>>> pixel_depth(width, std::vector<std::pair<float, int>>(height, { 1e9, -1 }));

    // 计算旋转矩阵，将点云坐标转换为图像平面坐标
    cv::Mat R;
    cv::Rodrigues(rvecsMat, R);

    // 第一次遍历，记录每个像素的最近点（只保留反射率>=10的点）
    for (size_t i = 0; i < pc.point_count; ++i) {
        const SimplePoint& pt = pc.points[i];
        if (pt.reflectivity < 0) continue; // 过滤低反射率点

        int u, v;
        if (point_to_pixel(pt, u, v, width, height)) {
            // 世界坐标转相机坐标
            cv::Mat pt3d = (cv::Mat_<double>(3, 1) << pt.x, pt.y, pt.z);
            cv::Mat pt_cam = R * pt3d + tvecsMat;
            float z_cam = static_cast<float>(pt_cam.at<double>(2, 0)); // 相机坐标系下的z

            if (z_cam < pixel_depth[u][v].first) {
                pixel_depth[u][v] = { z_cam, (int)i };
            }
        }
    }

    // 第二次遍历，锥体遮挡增强
    std::vector<ColoredPoint> colored_points;
    //•	点云密度高时，建议适当增大阈值，避免误判遮挡导致点云过于稀疏。
    //•	点云密度低时，建议减小阈值，避免过多保留被遮挡点，影响融合质量。
    //通过邻域深度比较判断点是否被遮挡
    float depth_threshold = 0.3f; // 阈值可根据实际点云密度调整

    for (int u = 0; u < width; ++u) {
        for (int v = 0; v < height; ++v) { 
            int idx = pixel_depth[u][v].second;
            if (idx >= 0) {
                float cur_depth = pixel_depth[u][v].first;
                bool occluded = false;
                // 检查3x3邻域，邻域越大考虑遮挡越多，点云越稀疏
                for (int du = -1; du <= 1 && !occluded; ++du) {
                    for (int dv = -1; dv <= 1 && !occluded; ++dv) {
                        if (du == 0 && dv == 0) continue;
                        int uu = u + du, vv = v + dv;
                        if (uu >= 0 && uu < width && vv >= 0 && vv < height) {
                            float neighbor_depth = pixel_depth[uu][vv].first;
                            // 如果邻域有更近的点，且深度差超过阈值，则认为当前点被遮挡
                            if (neighbor_depth < cur_depth - depth_threshold) {
                                occluded = true;
                            }
                        }
                    }
                }
                if (!occluded) {
                    const SimplePoint& pt = pc.points[idx];
                    cv::Vec3b color = image.at<cv::Vec3b>(v, u);
                    colored_points.push_back({ pt.x, pt.y, pt.z, color[2], color[1], color[0], pt.reflectivity });
                }
            }
        }
    }
    return colored_points;
}

// 6. 拼接PCD
void stitch_pcd(const std::vector<std::string>& files, const std::string& out_file) {
    using namespace std::chrono;
    auto start = high_resolution_clock::now();

    std::vector<ColoredPoint> all_points;
    for (const auto& f : files) {
        std::ifstream ifs(f);
        std::string line;
        // 跳过头部
        while (std::getline(ifs, line)) {
            if (line == "DATA ascii") break;
        }
        while (std::getline(ifs, line)) {
            std::istringstream iss(line);
            ColoredPoint pt;
            // 兼容两种格式：x y z r g b reflectivity 或 x y z rgb(float)
            if (line.find('.') != std::string::npos && std::count(line.begin(), line.end(), ' ') == 3) {
                // x y z rgb(float)
                float rgbf;
                iss >> pt.x >> pt.y >> pt.z >> rgbf;
                uint32_t rgb = *reinterpret_cast<uint32_t*>(&rgbf);
                pt.r = (rgb >> 16) & 0xFF;
                pt.g = (rgb >> 8) & 0xFF;
                pt.b = rgb & 0xFF;
                pt.reflectivity = 0;
            }
            else {
                int r, g, b, reflectivity;
                iss >> pt.x >> pt.y >> pt.z >> r >> g >> b >> reflectivity;
                pt.r = static_cast<uint8_t>(r);
                pt.g = static_cast<uint8_t>(g);
                pt.b = static_cast<uint8_t>(b);
                pt.reflectivity = static_cast<uint8_t>(reflectivity);
            }
            all_points.push_back(pt);
        }
    }

    auto end = high_resolution_clock::now();
    std::cout << "[串行拼接] 总耗时: " << duration_cast<milliseconds>(end - start).count() << " ms" << std::endl;

    // 存储为rgb float格式
    save_pcd_rgb(out_file, all_points);
}

void stitch_pcd_icp_parallel(const std::vector<std::string>& files, const std::string& out_file) {
    using namespace std::chrono;
    if (files.empty()) return;

    auto start = high_resolution_clock::now();

    // 假设每两个相邻帧一组，分配到多个线程
    size_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    size_t batch = (files.size() + num_threads - 1) / num_threads;

    std::vector<std::future<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>> futures;

    for (size_t t = 0; t < num_threads; ++t) {
        size_t begin = t * batch;
        size_t end = std::min(files.size(), begin + batch);
        if (begin >= end) break;

        futures.emplace_back(std::async(std::launch::async, [begin, end, &files]() {
            std::vector<ColoredPoint> ref_points = load_pcd(files[begin]);
            auto local_cloud = colored_points_to_pcl(ref_points);
            for (size_t i = begin + 1; i < end; ++i) {
                std::vector<ColoredPoint> cur_points = load_pcd(files[i]);
                auto cur_cloud = colored_points_to_pcl(cur_points);

                pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
                icp.setInputSource(cur_cloud);
                icp.setInputTarget(local_cloud);
                pcl::PointCloud<pcl::PointXYZRGB> aligned;
                icp.align(aligned);

                *local_cloud += aligned;
            }
            return local_cloud;
            }));
    }

    // 合并所有线程结果
    auto global_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>(); // 使用std::make_shared
    for (auto& fut : futures) {
        *global_cloud += *fut.get();
    }

    auto end = high_resolution_clock::now();
    std::cout << "[并行拼接] 总耗时: " << duration_cast<milliseconds>(end - start).count() << " ms" << std::endl;

    auto all_colored = pcl_to_colored_points(global_cloud);
    save_pcd_rgb(out_file, all_colored);
}

void merge_and_filter_clouds(const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& clouds, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& merged, float voxel_size = 0.01f) {
    for (const auto& cloud : clouds) {
        *merged += *cloud;
    }
    pcl::VoxelGrid<pcl::PointXYZRGB> voxel;
    voxel.setInputCloud(merged);
    voxel.setLeafSize(voxel_size, voxel_size, voxel_size);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
    voxel.filter(*filtered);
    merged.swap(filtered);
}

// 2. 定义兼容C回调的on_lidar_frame
void on_lidar_frame_callback(const LidarFrame* frame) {
    if (g_lidar_queue && g_lidar_mtx) {
        std::lock_guard<std::mutex> lock(*g_lidar_mtx);
        if (g_lidar_queue->size() < lidar_queue_max) {
            g_lidar_queue->push_back(*frame);
        }
    }
}

// 3. 修改collect_all
void collect_all(const std::string& rtsp_url, std::vector<ImageFrame>& images, std::vector<LidarFrame>& lidars) {

    std::cout << "[采集] 开始采集数据..." << std::endl;
    std::deque<ImageFrame> img_queue;
    std::deque<LidarFrame> lidar_queue;
    std::mutex img_mtx, lidar_mtx;
    std::atomic<bool> running{ true };

    // 设置全局指针
    g_lidar_queue = &lidar_queue;
    g_lidar_mtx = &lidar_mtx;

    std::thread img_thread(rtsp_reader, rtsp_url, std::ref(img_queue), std::ref(img_mtx), std::ref(running));
    lidar_core_init(on_lidar_frame_callback); // 只传递函数指针
    lidar_core_start();

    uint64_t first_ts = 0;
    while (running) {
        {
            std::lock_guard<std::mutex> lock(lidar_mtx);
            if (!lidar_queue.empty()) {
                if (!first_ts) first_ts = lidar_queue.front().timestamp;
                uint64_t last_ts = lidar_queue.back().timestamp;
                double time_span_sec = (last_ts - first_ts) / 1000.0;
                if (time_span_sec > SCAN_NUM) {
                    running = false;
                    break;
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    img_thread.join();
    lidar_core_stop();
    lidar_core_uninit();

    // 转存到vector
    {
        std::lock_guard<std::mutex> lock_img(img_mtx);
        std::lock_guard<std::mutex> lock_lidar(lidar_mtx);
        images.assign(img_queue.begin(), img_queue.end());
        lidars.assign(lidar_queue.begin(), lidar_queue.end());
    }
}
// 1. 融合所有点云与图像，只融合不保存
std::vector<FusedCloud> fuse_all_clouds(
    const std::vector<LidarFrame>& lidars,
    const std::vector<ImageFrame>& images)
{
    std::cout << "[融合] 开始融合点云与图像..." << std::endl;

    std::vector<FusedCloud> fused_clouds;
    for (const auto& pc : lidars) {
        // 匹配最近的图像
        size_t best_idx = 0;
        int64_t min_dt = std::numeric_limits<int64_t>::max();
        for (size_t i = 0; i < images.size(); ++i) {
            int64_t dt = std::abs((int64_t)images[i].timestamp - (int64_t)pc.timestamp);
            if (dt < min_dt) {
                min_dt = dt;
                best_idx = i;
            }
        }
        if (min_dt <= 1500 && !images.empty()) {
            auto fused_points = fuse_cloud_with_image(pc, images[best_idx].image);
            fused_clouds.push_back({ std::move(fused_points), images[best_idx].timestamp });
        }
    }
    return fused_clouds;
}

// 2. 拼接所有融合点云
std::vector<ColoredPoint> stitch_all_clouds(const std::vector<FusedCloud>& fused_clouds)
{
    std::cout << "[拼接] 开始拼接融合点云..." << std::endl;

    float theta = ROTATE_DEG_SEC * M_PI / 180.0f;
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clouds;
    for (size_t i = 0; i < fused_clouds.size(); ++i) {
        auto cloud = colored_points_to_pcl(fused_clouds[i].points);
        float angle = i * theta;
        Eigen::Matrix3f rot = Eigen::AngleAxisf(angle, Eigen::Vector3f::UnitZ()).toRotationMatrix();
        for (auto& pt : cloud->points) {
            Eigen::Vector3f p(pt.x, pt.y, pt.z);
            Eigen::Vector3f p_rot = rot * (p - SCAN_OFFSET) + SCAN_OFFSET;
            pt.x = p_rot.x();
            pt.y = p_rot.y();
            pt.z = p_rot.z();
        }
        clouds.push_back(cloud);
    }
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr merged(new pcl::PointCloud<pcl::PointXYZRGB>);
    merge_and_filter_clouds(clouds, merged, 0.01f);
    return pcl_to_colored_points(merged);
}

// 修改save_all，所有文件名用年月日时分秒
void save_all(
    const std::vector<ImageFrame>& images,
    const std::vector<LidarFrame>& lidars,
    const std::vector<FusedCloud>& fused_clouds,
    const std::vector<ColoredPoint>& stitched)
{
    std::cout << "[存储] 开始保存所有数据..." << std::endl;
    // 创建数据目录
    namespace fs = std::filesystem;
    fs::create_directories("data");
    fs::create_directories("data/fused");

    // 保存拼接点云
    save_pcd_rgb("stitched.pcd", stitched);

    // 保存融合点云
    for (const auto& fused : fused_clouds) {
        std::string fused_pcd_name = "data/fused/" + format_datetime(fused.timestamp) + "_fused.pcd";
        save_pcd_rgb(fused_pcd_name, fused.points);
    }

    // 保存原始图像
    for (const auto& img : images) {
        std::string jpg_name = "data/" + format_datetime(img.timestamp) + ".jpg";
        cv::imwrite(jpg_name, img.image);
    }

    // 保存原始点云
    for (const auto& pc : lidars) {
        std::string pcd_name = "data/" + format_datetime(pc.timestamp) + ".pcd";
        std::vector<ColoredPoint> colored_points;
        for (size_t i = 0; i < pc.point_count; ++i) {
            const SimplePoint& pt = pc.points[i];
            colored_points.push_back({ pt.x, pt.y, pt.z, 0, 0, 0, pt.reflectivity });
        }
        save_pcd(pcd_name, colored_points);
    }
}

void realtime_show_fusion_cloud(const std::vector<FusedCloud>& fused_clouds) {
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Realtime Fusion Cloud (Memory)"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->setSize(1280, 720);
    viewer->initCameraParameters();

    size_t last_idx = static_cast<size_t>(-1);
    size_t shown_count = 0;
    while (!viewer->wasStopped()) {
        // 找到最新的融合点云
        if (!fused_clouds.empty()) {
            size_t latest_idx = shown_count;
            if (latest_idx < fused_clouds.size()) {
                const auto& pts = fused_clouds[latest_idx].points;
                auto cloud = colored_points_to_pcl(pts);
                viewer->removeAllPointClouds();
                viewer->addPointCloud<pcl::PointXYZRGB>(cloud, "fusion_cloud");
                viewer->setCameraPosition(
                    0, 0, 0,    // 相机位置
                    1, 0, 0,    // 看向x正方向
                    0, 0, 1,    // z轴朝上
                    0.1
                );
                last_idx = latest_idx;
                std::cout << "显示融合点云: " << format_datetime(fused_clouds[latest_idx].timestamp) << std::endl;
                ++shown_count;
            }
            else {
                // 所有融合点云已显示，自动关闭窗口
                viewer->close();
                break;
            }
        }
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}


void stitch_pcd_with_rotation_parallel(const std::vector<std::string>& files, const std::string& out_file, const Eigen::Vector3f& offset) {
    if (files.empty()) return;

    // 1. 解析每个文件的时间戳并读取点云
    std::vector<std::pair<std::string, uint64_t>> file_time;
    std::vector<std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>>> loaded_clouds;
    auto parse_timestamp = [](const std::string& name) -> uint64_t {
        std::tm tm = {};
        int ms = 0;
        if (sscanf(name.c_str(), "%4d%2d%2d_%2d%2d%2d_%3d",
            &tm.tm_year, &tm.tm_mon, &tm.tm_mday,
            &tm.tm_hour, &tm.tm_min, &tm.tm_sec, &ms) == 7) {
            tm.tm_year -= 1900;
            tm.tm_mon -= 1;
            uint64_t ts = static_cast<uint64_t>(mktime(&tm)) * 1000 + ms;
            return ts;
        }
        return 0;
        };
    for (const auto& f : files) {
        std::string stem = std::filesystem::path(f).stem().string();
        file_time.emplace_back(f, parse_timestamp(stem));
        loaded_clouds.push_back(colored_points_to_pcl(load_pcd_rgb(f)));
    }
    std::vector<size_t> idx(file_time.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return file_time[a].second < file_time[b].second; });
    if (idx.size() < 2) return;
    uint64_t t0 = file_time[idx[0]].second;

    using namespace std::chrono;
    auto start = high_resolution_clock::now();

    // 2. 并行处理
    size_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    size_t batch = (idx.size() + num_threads - 1) / num_threads;
    std::vector<std::future<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>> futures;

    for (size_t t = 0; t < num_threads; ++t) {
        size_t begin = t * batch;
        size_t end = std::min(idx.size(), begin + batch);
        if (begin >= end) break;
        futures.emplace_back(std::async(std::launch::async, [begin, end, &idx, &file_time, &loaded_clouds, t0, offset]() {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr merged(new pcl::PointCloud<pcl::PointXYZRGB>);
            for (size_t k = begin; k < end; ++k) {
                size_t i = idx[k];
                auto cloud = loaded_clouds[i];
                double dt = (file_time[i].second - t0) / 1000.0;
                double angle = ROTATE_DEG_SEC * dt * M_PI / 180.0;
                Eigen::Matrix3f rot = Eigen::AngleAxisf(angle, Eigen::Vector3f::UnitZ()).toRotationMatrix();
                for (auto& pt : cloud->points) {
                    Eigen::Vector3f p(pt.x, pt.y, pt.z);
                    Eigen::Vector3f p_rot = rot * (p - offset) + offset;
                    pt.x = p_rot.x();
                    pt.y = p_rot.y();
                    pt.z = p_rot.z();
                }
                *merged += *cloud;
            }
            return merged;
            }));
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr merged(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (auto& fut : futures) {
        *merged += *fut.get();
    }

    // 体素滤波
    pcl::VoxelGrid<pcl::PointXYZRGB> voxel;
    voxel.setInputCloud(merged);
    voxel.setLeafSize(0.01f, 0.01f, 0.01f);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
    voxel.filter(*filtered);

    auto end = high_resolution_clock::now();
    std::cout << "[并行旋转拼接-仅运算] 总耗时: " << duration_cast<milliseconds>(end - start).count() << " ms" << std::endl;

    pcl::io::savePCDFileBinary(out_file, *filtered);
}
void stitch_pcd_with_rotation(const std::vector<std::string>& files, const std::string& out_file, const Eigen::Vector3f& offset) {
    if (files.empty()) return;
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clouds;
    float theta = ROTATE_DEG_SEC * M_PI / 180.0f;

    // 读取所有点云
    std::vector<std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>>> loaded_clouds;
    for (const auto& f : files) {
        auto cloud = colored_points_to_pcl(load_pcd_rgb(f));
        loaded_clouds.push_back(cloud);
    }

    using namespace std::chrono;
    auto start = high_resolution_clock::now();

    // 拼接与旋转
    // 第一帧无需旋转
    clouds.push_back(loaded_clouds[0]);
    for (size_t i = 1; i < loaded_clouds.size(); ++i) {
        auto cloud = loaded_clouds[i];
        float angle = i * theta;
        Eigen::Matrix3f rot = Eigen::AngleAxisf(angle, Eigen::Vector3f::UnitZ()).toRotationMatrix();
        for (auto& pt : cloud->points) {
            Eigen::Vector3f p(pt.x, pt.y, pt.z);
            Eigen::Vector3f p_rot = rot * (p - offset) + offset;
            //Eigen::Vector3f p_rot = rot * p;
            pt.x = p_rot.x();
            pt.y = p_rot.y();
            pt.z = p_rot.z();
        }
        clouds.push_back(cloud);
    }
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr merged(new pcl::PointCloud<pcl::PointXYZRGB>);
    merge_and_filter_clouds(clouds, merged, 0.01f);

    auto end = high_resolution_clock::now();
    std::cout << "[串行旋转拼接-仅运算] 总耗时: " << duration_cast<milliseconds>(end - start).count() << " ms" << std::endl;

    // 存储
    auto all_colored = pcl_to_colored_points(merged);
    save_pcd_rgb(out_file, all_colored);
}

void realtime_show_stitching(const std::vector<std::string>& files, const Eigen::Vector3f& offset, const std::string& image_dir = "stitching_images") {
    if (files.empty()) return;
    float theta = ROTATE_DEG_SEC * M_PI / 180.0f;

    // 创建图片输出目录
    namespace fs = std::filesystem;
    fs::create_directories(image_dir);

    // 读取所有点云
    std::vector<std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>>> loaded_clouds;
    for (const auto& f : files) {
        auto cloud = colored_points_to_pcl(load_pcd_rgb(f));
        loaded_clouds.push_back(cloud);
    }

    // 初始化可视化窗口
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Stitching Realtime"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->setSize(1280, 720);
    viewer->initCameraParameters();

    int frame_width = 1280, frame_height = 720;

    // 累积点云
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr merged(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (size_t i = 0; i < loaded_clouds.size(); ++i) {
        auto cloud = loaded_clouds[i];
        float angle = i * theta;
        Eigen::Matrix3f rot = Eigen::AngleAxisf(angle, Eigen::Vector3f::UnitZ()).toRotationMatrix();
        for (auto& pt : cloud->points) {
            Eigen::Vector3f p(pt.x, pt.y, pt.z);
            Eigen::Vector3f p_rot = rot * (p - offset) + offset;
            pt.x = p_rot.x();
            pt.y = p_rot.y();
            pt.z = p_rot.z();
        }
        *merged += *cloud;
    
        // 实时显示
        viewer->removeAllPointClouds();
        viewer->addPointCloud<pcl::PointXYZRGB>(merged, "stitched_cloud");
        viewer->setCameraPosition(
            0, 0, 0,         // 相机位置
            cos(SCAN_ORDER * 45.0 * M_PI / 180.0), sin(SCAN_ORDER * 45.0 * M_PI / 180.0), 0, // 视点（看向左侧60度）
            0, 0, 1,          // 上方向
            0.1
        );
        viewer->spinOnce(100);
        std::cout << "已拼接帧数: " << (i + 1) << "/" << loaded_clouds.size() << std::endl;

        // 截图并保存为JPG
        auto img = viewer->getRenderWindow()->GetRGBACharPixelData(0, 0, frame_width - 1, frame_height - 1, 1);
        cv::Mat frame(frame_height, frame_width, CV_8UC4, img);
        cv::Mat bgr;
        cv::cvtColor(frame, bgr, cv::COLOR_RGBA2BGR);
        std::string img_name = image_dir + "/stitch_" + std::to_string(i + 1) + ".png";
        cv::imwrite(img_name, bgr);
        delete[] img;

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        if (viewer->wasStopped()) break;
    }
    viewer->close();
}

void realtime_show_stitching_cloud(const std::vector<FusedCloud>& fused_clouds, const Eigen::Vector3f& offset) {
    if (fused_clouds.empty()) return;
    float theta = ROTATE_DEG_SEC * M_PI / 180.0f;

    // 初始化可视化窗口
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Stitching Realtime (Memory)"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->setSize(1280, 720);
    viewer->initCameraParameters();

    // 累积点云
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr merged(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (size_t i = 0; i < fused_clouds.size(); ++i) {
        auto cloud = colored_points_to_pcl(fused_clouds[i].points);
        float angle = i * theta;
        Eigen::Matrix3f rot = Eigen::AngleAxisf(angle, Eigen::Vector3f::UnitZ()).toRotationMatrix();
        for (auto& pt : cloud->points) {
            Eigen::Vector3f p(pt.x, pt.y, pt.z);
            Eigen::Vector3f p_rot = rot * (p - offset) + offset;
            pt.x = p_rot.x();
            pt.y = p_rot.y();
            pt.z = p_rot.z();
        }
        *merged += *cloud;

        // 实时显示
        viewer->removeAllPointClouds();
        viewer->addPointCloud<pcl::PointXYZRGB>(merged, "stitched_cloud");
        viewer->setCameraPosition(
            0, 0, 0,
            cos(SCAN_ORDER * 45.0 * M_PI / 180.0), sin(SCAN_ORDER * 45.0 * M_PI / 180.0), 0,
            0, 0, 1,
            0.1
        );
        viewer->spinOnce(100);
        std::cout << "已拼接帧数: " << (i + 1) << "/" << fused_clouds.size() << std::endl;

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        if (viewer->wasStopped()) break;
    }
    viewer->close();
}

void process_stitching_images(const std::string& src_dir, const std::string& dst_dir) {
    namespace fs = std::filesystem;
    fs::create_directories(dst_dir);

    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(src_dir)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());

    for (const auto& file : files) {
        cv::Mat img = cv::imread(file.string(), cv::IMREAD_COLOR);
        if (img.empty()) continue;

        int min_x = img.cols, min_y = img.rows, max_x = -1, max_y = -1;
        // 查找第一个和最后一个黑色像素
        for (int y = 0; y < img.rows; ++y) {
            for (int x = 0; x < img.cols; ++x) {
                const cv::Vec3b& pix = img.at<cv::Vec3b>(y, x);
                if (pix == cv::Vec3b(0, 0, 0)) {
                    if (x < min_x) min_x = x;
                    if (x > max_x) max_x = x;
                    if (y < min_y) min_y = y;
                    if (y > max_y) max_y = y;
                }
            }
        }
        // 若没有黑色像素则跳过
        if (min_x > max_x || min_y > max_y) continue;

        // 裁剪
        cv::Rect roi(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1);
        cv::Mat cropped = img(roi);

        // 上下翻转（绕X轴旋转180度）
        cv::Mat flipped;
        cv::flip(cropped, flipped, 0);

        // 保存
        std::string out_name = dst_dir + "/" + file.filename().string();
        std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, 98 };
        cv::imwrite(out_name, flipped, params);
    }
}

int main() {
    Eigen::Vector3f offset(-0.0885f, 0.0435f, -0.076f);
    // 单独测试拼接
    bool test = true;

    if (test) {
        std::cout << "测试拼接功能..." << std::endl;
        std::vector<std::string> fused_pcd_files;
        namespace fs = std::filesystem;
        fused_pcd_files.clear();
        if (fs::exists("data/fused")) {
            for (const auto& entry : fs::directory_iterator("data/fused")) {
                if (entry.path().extension() == ".pcd") {
                    fused_pcd_files.push_back(entry.path().string());
                }
            }
        }
        while(1)
        // 实时显示拼接过程
            realtime_show_stitching(fused_pcd_files, offset, "data/stitching_images");
        //process_stitching_images("data/stitching_images", "data/stitching_images_processed");
        // 选择使用串行或并行拼接
        //stitch_pcd_with_rotation(fused_pcd_files, "stitched_all.pcd", offset);
        //stitch_pcd_with_rotation_parallel(fused_pcd_files, "stitched_parallel.pcd", offset);
        return 0;
    }
    else
    {
        std::string rtsp_url = "rtsp://admin:123456@192.168.1.108:554/h265/ch1/main/av_stream";

        // 采集
        std::vector<ImageFrame> images;
        std::vector<LidarFrame> lidars;
        std::cout << "数据采集和处理开始！" << std::endl;
        collect_all(rtsp_url, images, lidars);

        // 融合
        auto fused_clouds = fuse_all_clouds(lidars, images);

        // 实时显示融合点云
        realtime_show_fusion_cloud(fused_clouds);

        // 拼接
        auto stitched = stitch_all_clouds(fused_clouds);

        // 存储
        save_all(images, lidars, fused_clouds, stitched);
        std::cout << "数据采集和处理完成！" << std::endl;

        // 拼接显示
        std::cout << "连续拼接显示..." << std::endl;
        std::vector<std::string> fused_pcd_files;
        namespace fs = std::filesystem;
        fused_pcd_files.clear();
        if (fs::exists("data/fused")) {
            for (const auto& entry : fs::directory_iterator("data/fused")) {
                if (entry.path().extension() == ".pcd") {
                    fused_pcd_files.push_back(entry.path().string());
                }
            }
        }
        // 实时显示拼接过程
        //realtime_show_stitching(fused_pcd_files, offset, "data/stitching_images");
        realtime_show_stitching_cloud(fused_clouds, offset);
        //process_stitching_images("data/stitching_images", "data/stitching_images_processed");
        // 选择使用串行或并行拼接
        //stitch_pcd_with_rotation(fused_pcd_files, "stitched_all.pcd", offset);

        return 0;
    }
}
