#include <chrono>
#include <iostream>

#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

namespace {

template <typename T = double>
class Timer {
  std::chrono::system_clock::time_point start_t_, end_t_;
  T elapsed_msec_{-1};
  size_t history_num_{30};
  std::vector<T> history_;

 public:
  Timer() {}
  ~Timer() {}
  explicit Timer(size_t history_num) : history_num_(history_num) {}

  std::chrono::system_clock::time_point start_t() const { return start_t_; }
  std::chrono::system_clock::time_point end_t() const { return end_t_; }

  void Start() { start_t_ = std::chrono::system_clock::now(); }
  void End() {
    end_t_ = std::chrono::system_clock::now();
    elapsed_msec_ = static_cast<T>(
        std::chrono::duration_cast<std::chrono::microseconds>(end_t_ - start_t_)
            .count() *
        0.001);

    history_.push_back(elapsed_msec_);
    if (history_num_ < history_.size()) {
      history_.erase(history_.begin());
    }
  }
  T elapsed_msec() const { return elapsed_msec_; }
  T average_msec() const {
    if (history_.empty()) {
      return elapsed_msec_;
    }
    return static_cast<T>(
        std::accumulate(history_.begin(), history_.end(), T(0)) /
        history_.size());
  }
};

void Undistort(const cv::Mat& src, cv::Mat& dst, const cv::Mat& _cameraMatrix,
               const cv::Mat& _distCoeffs, const cv::Mat& _newCameraMatrix,
               int interpolation = cv::INTER_LINEAR,
               int borderMode = cv::BORDER_CONSTANT,
               const cv::Scalar& borderValue = cv::Scalar()) {
  using namespace cv;
  dst.create(src.size(), src.type());
  CV_Assert(dst.data != src.data);

  int stripe_size0 =
      std::min(std::max(1, (1 << 12) / std::max(src.cols, 1)), src.rows);
  Mat map1(stripe_size0, src.cols, CV_16SC2),
      map2(stripe_size0, src.cols, CV_16UC1);

  Mat_<double> A, distCoeffs, Ar, I = Mat_<double>::eye(3, 3);

  _cameraMatrix.convertTo(A, CV_64F);
  if (_distCoeffs.data)
    distCoeffs = Mat_<double>(_distCoeffs);
  else {
    distCoeffs.create(5, 1);
    distCoeffs = 0.;
  }

  if (_newCameraMatrix.data)
    _newCameraMatrix.convertTo(Ar, CV_64F);
  else
    A.copyTo(Ar);

  double v0 = Ar(1, 2);
  for (int y = 0; y < src.rows; y += stripe_size0) {
    int stripe_size = std::min(stripe_size0, src.rows - y);
    Ar(1, 2) = v0 - y;
    Mat map1_part = map1.rowRange(0, stripe_size),
        map2_part = map2.rowRange(0, stripe_size),
        dst_part = dst.rowRange(y, y + stripe_size);

    initUndistortRectifyMap(A, distCoeffs, I, Ar, Size(src.cols, stripe_size),
                            map1_part.type(), map1_part, map2_part);
    remap(src, dst_part, map1_part, map2_part, interpolation, borderMode,
          borderValue);
  }
}

std::vector<std::pair<cv::Mat, cv::Mat>> PrepareUndistortRectifyMap(
    const cv::Mat& src, const cv::Mat& _cameraMatrix,
    const cv::Mat& _distCoeffs, const cv::Mat& _newCameraMatrix) {
  using namespace cv;

  int stripe_size0 =
      std::min(std::max(1, (1 << 12) / std::max(src.cols, 1)), src.rows);
  Mat map1(stripe_size0, src.cols, CV_16SC2),
      map2(stripe_size0, src.cols, CV_16UC1);

  Mat_<double> A, distCoeffs, Ar, I = Mat_<double>::eye(3, 3);

  _cameraMatrix.convertTo(A, CV_64F);
  if (_distCoeffs.data)
    distCoeffs = Mat_<double>(_distCoeffs);
  else {
    distCoeffs.create(5, 1);
    distCoeffs = 0.;
  }

  if (_newCameraMatrix.data)
    _newCameraMatrix.convertTo(Ar, CV_64F);
  else
    A.copyTo(Ar);

  double v0 = Ar(1, 2);
  std::vector<std::pair<Mat, Mat>> map_parts;
  for (int y = 0; y < src.rows; y += stripe_size0) {
    int stripe_size = std::min(stripe_size0, src.rows - y);
    Ar(1, 2) = v0 - y;
    Mat map1_part = map1.rowRange(0, stripe_size),
        map2_part = map2.rowRange(0, stripe_size);
    initUndistortRectifyMap(A, distCoeffs, I, Ar, Size(src.cols, stripe_size),
                            map1_part.type(), map1_part, map2_part);
    // Need to clone() because rowRange() does not copy data
    map_parts.push_back({map1_part.clone(), map2_part.clone()});
  }
  return map_parts;
}

void ApplyUndistortRectifyMap(
    const cv::Mat& src, cv::Mat& dst,
    const std::vector<std::pair<cv::Mat, cv::Mat>>& map_parts,
    int interpolation = cv::INTER_LINEAR, int borderMode = cv::BORDER_CONSTANT,
    const cv::Scalar& borderValue = cv::Scalar()) {
  using namespace cv;
  if (dst.size() != src.size() || dst.type() != src.type()) {
    dst.create(src.size(), src.type());
  }
  CV_Assert(dst.data != src.data);

  int stripe_size0 =
      std::min(std::max(1, (1 << 12) / std::max(src.cols, 1)), src.rows);

  int count = 0;
  for (int y = 0; y < src.rows; y += stripe_size0) {
    int stripe_size = std::min(stripe_size0, src.rows - y);
    Mat map1_part = map_parts[count].first, map2_part = map_parts[count].second,
        dst_part = dst.rowRange(y, y + stripe_size);
    remap(src, dst_part, map1_part, map2_part, interpolation, borderMode,
          borderValue);
    count++;
  }
}
}  // namespace

int main() {
  auto depth = cv::imread("../data/cmu_panoptic/171026_cello3/depth_00000.png",
                          cv::ImreadModes::IMREAD_UNCHANGED);

  double d_cx = 255.8067652;
  double d_cy = 209.4778628;
  double d_fx = 364.7317041;
  double d_fy = 364.7317041;
  std::vector<double> d_ks = {0.09561943152, -0.2719577089, 0.09246527415};

  double d_p1 = 9.653866458e-05;
  double d_p2 = -0.0001325012348;

  cv::Mat1d d_cammat = cv::Mat1d::eye(3, 3);
  d_cammat.at<double>(0, 0) = d_fx;
  d_cammat.at<double>(1, 1) = d_fy;
  d_cammat.at<double>(0, 2) = d_cx;
  d_cammat.at<double>(1, 2) = d_cy;

  cv::Mat1d d_distcoeffs = cv::Mat1d::zeros(5, 1);
  d_distcoeffs.at<double>(0, 0) = d_ks[0];
  d_distcoeffs.at<double>(1, 0) = d_ks[1];
  d_distcoeffs.at<double>(2, 0) = d_p1;
  d_distcoeffs.at<double>(3, 0) = d_p2;
  d_distcoeffs.at<double>(4, 0) = d_ks[2];
  // d_distcoeffs.at<double>(5, 0) = d_ks[3];
  // d_distcoeffs.at<double>(6, 0) = d_ks[4];
  // d_distcoeffs.at<double>(7, 0) = d_ks[5];

  Timer<> timer;

  cv::Mat depth_out;
  int max_loop = 100;
  timer.Start();
  for (int i = 0; i < max_loop; i++) {
    depth_out = cv::Mat();
    cv::undistort(depth, depth_out, d_cammat, d_distcoeffs, cv::Mat());
  }
  timer.End();
  std::cout << "cv::undistort " << timer.elapsed_msec() / max_loop << std::endl;
  cv::imwrite("cv_undistort.png", depth_out);

  timer.Start();
  for (int i = 0; i < max_loop; i++) {
    depth_out = cv::Mat();
    Undistort(depth, depth_out, d_cammat, d_distcoeffs, cv::Mat());
  }
  timer.End();
  std::cout << "Undistort " << timer.elapsed_msec() / max_loop << std::endl;
  cv::imwrite("Undistort.png", depth_out);

  timer.Start();
  for (int i = 0; i < max_loop; i++) {
    depth_out = cv::Mat();
    Undistort(depth, depth_out, d_cammat, d_distcoeffs, cv::Mat(),
              cv::InterpolationFlags::INTER_NEAREST);
  }
  timer.End();
  std::cout << "Undistort NN " << timer.elapsed_msec() / max_loop << std::endl;
  cv::imwrite("Undistort_NN.png", depth_out);

  timer.Start();
  auto map_parts =
      PrepareUndistortRectifyMap(depth, d_cammat, d_distcoeffs, cv::Mat());
  timer.End();
  std::cout << "PrepareUndistortRectifyMap " << timer.elapsed_msec()
            << std::endl;
  for (int i = 0; i < max_loop; i++) {
    depth_out = cv::Mat();
    ApplyUndistortRectifyMap(depth, depth_out, map_parts,
                             cv::InterpolationFlags::INTER_NEAREST);
  }
  timer.End();
  std::cout << "ApplyUndistortRectifyMap NN " << timer.elapsed_msec() / max_loop
            << std::endl;
  cv::imwrite("Undistort_NN2.png", depth_out);

  return 0;
}