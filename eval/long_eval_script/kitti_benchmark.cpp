// BSD 3-Clause License

// Copyright (c) 2024, Robots Vision and Perception Group

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.

// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.

// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// KITTI Odometry Benchmark Evaluation Tool
// Adapted from VBR benchmark for KITTI dataset format

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <limits>
#include <algorithm>
#include <Eigen/Dense>
#include <iomanip>
#include <sstream>
#include <map>

// KITTI standard evaluation lengths (in meters)
const std::vector<float> LENGTHS = {100, 200, 300, 400, 500, 600, 700, 800};

// KITTI odometry sequences (00-10 have ground truth)
const std::vector<std::string> SEQ_NAMES = {
    "00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"
};

struct Error
{
  int first_frame;
  double r_err, t_err;
  float len;
  Error(int first_frame, double r_err, double t_err, float len) : first_frame(first_frame), r_err(r_err), t_err(t_err), len(len) {}
};

struct ErrorPair
{
  double r_err = 0.;
  double t_err = 0.;
  ErrorPair() = default;
  ErrorPair(double r_err, double t_err) : r_err(r_err), t_err(t_err) {}

  void operator+=(const ErrorPair& e) {
    r_err += e.r_err;
    t_err += e.t_err;
  } 
};

struct Stats
{
  std::string sequence_name_;
  double r_err, t_err;
  Stats(std::string sequence_name, double r_err, double t_err) : sequence_name_(sequence_name), r_err(r_err), t_err(t_err) {}
};

struct Pose
{
  int frame_id;  // Frame index for KITTI (no timestamps in GT)
  Eigen::Isometry3f transform;
  Pose(int frame_id, Eigen::Isometry3f transform) : frame_id(frame_id), transform(transform) {}
};

inline bool sortComparator(const Stats &stat_l, const Stats &stat_r)
{
  return stat_l.t_err < stat_r.t_err;
}

// Load KITTI GT poses (3x4 matrix format, no timestamps)
// Format: r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
inline std::vector<Pose> loadKittiGTPoses(const std::string &file_name)
{
  std::vector<Pose> poses;
  std::ifstream file(file_name);
  if (!file.is_open())
  {
    std::cout << "error: unable to open GT file " << file_name << std::endl;
    return poses;
  }

  int frame_id = 0;
  std::string line;
  while (std::getline(file, line))
  {
    if (line.empty() || line[0] == '#')
      continue;

    std::istringstream iss(line);
    double r11, r12, r13, tx, r21, r22, r23, ty, r31, r32, r33, tz;
    if (!(iss >> r11 >> r12 >> r13 >> tx >> r21 >> r22 >> r23 >> ty >> r31 >> r32 >> r33 >> tz))
    {
      std::cerr << "error reading line from GT file: " << line << std::endl;
      continue;
    }

    Eigen::Isometry3f P = Eigen::Isometry3f::Identity();
    Eigen::Matrix3f R;
    R << r11, r12, r13,
         r21, r22, r23,
         r31, r32, r33;
    P.linear() = R;
    P.translation() << tx, ty, tz;

    poses.push_back(Pose(frame_id, P));
    frame_id++;
  }

  file.close();
  return poses;
}

// Load estimated poses (TUM format with timestamp: t x y z qx qy qz qw)
inline std::vector<Pose> loadEstimatedPoses(const std::string &file_name)
{
  std::vector<Pose> poses;
  std::ifstream file(file_name);
  if (!file.is_open())
  {
    std::cout << "error: unable to open estimated file " << file_name << std::endl;
    return poses;
  }

  std::string line;
  while (std::getline(file, line))
  {
    if (line.empty() || line[0] == '#')
      continue;

    std::istringstream iss(line);
    float t, x, y, z, qx, qy, qz, qw;
    if (!(iss >> t >> x >> y >> z >> qx >> qy >> qz >> qw))
    {
      std::cerr << "error reading line from estimated file: " << line << std::endl;
      continue;
    }

    Eigen::Isometry3f P = Eigen::Isometry3f::Identity();
    P.translation() << x, y, z;

    const Eigen::Quaternionf q(qw, qx, qy, qz);
    P.linear() = q.toRotationMatrix();

    // Use timestamp as frame_id (assuming integer frame indices)
    poses.push_back(Pose(static_cast<int>(t), P));
  }

  file.close();
  return poses;
}

// Match estimated poses to GT poses by frame index
inline std::pair<std::vector<Pose>, std::vector<Pose>> matchByFrameIndex(
    const std::vector<Pose> &poses_gt, const std::vector<Pose> &poses_es)
{
  std::vector<Pose> poses_gt_matched;
  std::vector<Pose> poses_es_matched;

  // Build a map from frame_id to estimated pose index
  std::map<int, size_t> es_frame_map;
  for (size_t i = 0; i < poses_es.size(); ++i)
  {
    es_frame_map[poses_es[i].frame_id] = i;
  }

  // Match GT poses with estimated poses
  for (const auto &pose_gt : poses_gt)
  {
    auto it = es_frame_map.find(pose_gt.frame_id);
    if (it != es_frame_map.end())
    {
      poses_gt_matched.push_back(pose_gt);
      poses_es_matched.push_back(poses_es[it->second]);
    }
  }

  return std::make_pair(poses_gt_matched, poses_es_matched);
}

inline std::vector<float> trajectoryDistances(const std::vector<Pose> &poses)
{
  std::vector<float> dist;
  dist.push_back(0);
  for (size_t i = 1; i < poses.size(); ++i)
  {
    const Eigen::Vector3f t1 = poses[i - 1].transform.translation();
    const Eigen::Vector3f t2 = poses[i].transform.translation();

    dist.push_back(dist[i - 1] + (t1 - t2).norm());
  }
  return dist;
}

inline size_t lastFrameFromSegmentLength(const std::vector<float> &dist, const int &first_frame, const float &len)
{
  for (size_t i = first_frame; i < dist.size(); ++i)
    if (dist[i] > dist[first_frame] + len)
      return i;
  return -1;
}

inline double rotationError(const Eigen::Isometry3f &pose_error)
{
  Eigen::Quaternionf q(pose_error.linear());
  q.normalize();

  const Eigen::Quaternionf q_identity(1.0f, 0.0f, 0.0f, 0.0f);
  const double error_radians = q_identity.angularDistance(q);

  const double error_degrees = error_radians * (180.0f / M_PI);
  return error_degrees;
}

inline double translationError(const Eigen::Isometry3f &pose_error)
{
  const Eigen::Vector3f t = pose_error.translation();
  return t.norm();
}

inline std::vector<Error> computeSequenceErrors(const std::vector<Pose> poses_gt, const std::vector<Pose> &poses_es)
{
  std::vector<Error> err;

  const std::vector<float> dist = trajectoryDistances(poses_gt);
  const float seq_length = dist.back();
  std::cout << "sequence length [m]: " << seq_length << std::endl;

  // Use standard KITTI lengths
  std::cout << "using KITTI standard lengths: ";
  for (const float &len : LENGTHS)
  {
    std::cout << len << "m ";
  }
  std::cout << std::endl << std::endl;

  for (size_t first_frame = 0; first_frame < poses_gt.size(); ++first_frame)
  {
    for (size_t i = 0; i < LENGTHS.size(); ++i)
    {
      const float curr_len = LENGTHS[i];
      const int last_frame = lastFrameFromSegmentLength(dist, first_frame, curr_len);

      if (last_frame == -1)
        continue;

      const Eigen::Isometry3f pose_delta_gt = poses_gt[first_frame].transform.inverse() * poses_gt[last_frame].transform;
      const Eigen::Isometry3f pose_delta_es = poses_es[first_frame].transform.inverse() * poses_es[last_frame].transform;
      const Eigen::Isometry3f pose_error = pose_delta_es.inverse() * pose_delta_gt;
      const double r_err = rotationError(pose_error);
      const double t_err = translationError(pose_error);

      err.push_back(Error(first_frame, r_err / curr_len, t_err / curr_len, curr_len));
    }
  }

  return err;
}

inline std::vector<Pose> computeAlignedEstimate(const std::vector<Pose> &poses_gt, const std::vector<Pose> &poses_es, bool use_sim3 = true)
{
  std::vector<Pose> poses_es_aligned;
  poses_es_aligned.reserve(poses_es.size());

  Eigen::Matrix<float, 3, Eigen::Dynamic> gt_matrix;
  gt_matrix.resize(Eigen::NoChange, poses_gt.size());
  for (size_t i = 0; i < poses_gt.size(); ++i)
    gt_matrix.col(i) = poses_gt[i].transform.translation();

  Eigen::Matrix<float, 3, Eigen::Dynamic> es_matrix;
  es_matrix.resize(Eigen::NoChange, poses_es.size());
  for (size_t i = 0; i < poses_es.size(); ++i)
    es_matrix.col(i) = poses_es[i].transform.translation();

  // use_sim3=true for monocular (Sim3), false for stereo/LiDAR (SE3)
  const Eigen::Matrix4f transform_matrix = Eigen::umeyama(es_matrix, gt_matrix, use_sim3);
  Eigen::Isometry3f transform = Eigen::Isometry3f(transform_matrix.block<3, 3>(0, 0));
  transform.translation() = transform_matrix.block<3, 1>(0, 3);

  for (size_t i = 0; i < poses_es.size(); ++i)
    poses_es_aligned.push_back(Pose(poses_es[i].frame_id, transform * poses_es[i].transform));

  return poses_es_aligned;
}

inline std::pair<Stats, std::vector<ErrorPair>> computeSequenceRPE(const std::vector<Error> &seq_err, const std::string &sequence_name, size_t num_poses)
{
  double t_err = 0;
  double r_err = 0;

  std::vector<ErrorPair> RPE_errors;
  RPE_errors.resize(num_poses);
  std::vector<int> count;
  count.resize(num_poses);

  for (const Error &error : seq_err)
  {
    RPE_errors[error.first_frame] += ErrorPair(error.r_err, error.t_err);
    count[error.first_frame]++;
    t_err += error.t_err;
    r_err += error.r_err;
  }

  for (size_t i = 0; i < num_poses; ++i) {
    if(!count[i])
      continue;
    RPE_errors[i].r_err /= count[i];
    RPE_errors[i].t_err /= count[i];
  }

  const double r_rpe = r_err / double(seq_err.size());
  const double t_rpe = 100 * t_err / double(seq_err.size());
  return std::make_pair(Stats(sequence_name, r_rpe, t_rpe), RPE_errors);
}

inline std::pair<Stats, std::vector<ErrorPair>> computeSequenceATE(const std::vector<Pose> &poses_gt, const std::vector<Pose> &poses_es_aligned, const std::string &sequence_name)
{
  double r_sum = 0;
  double t_sum = 0;
  std::vector<ErrorPair> ATE_errors;
  ATE_errors.reserve(poses_gt.size());

  for (size_t i = 0; i < poses_gt.size(); ++i)
  {
    const Eigen::Isometry3f pose_error = poses_gt[i].transform.inverse() * poses_es_aligned[i].transform;
    const double r_err = rotationError(pose_error);
    const double t_err = translationError(pose_error);

    ATE_errors.push_back(ErrorPair(r_err, t_err));

    r_sum += r_err * r_err;  // Square for RMSE
    t_sum += t_err * t_err;
  }

  const double r_ate_rmse = std::sqrt(r_sum / double(poses_gt.size()));
  const double t_ate_rmse = std::sqrt(t_sum / double(poses_gt.size()));
  return std::make_pair(Stats(sequence_name, r_ate_rmse, t_ate_rmse), ATE_errors);
}

inline void computeAndPrintStats(std::vector<Stats> &stats, const std::string &path_to_result_file)
{
  FILE *fp = fopen(path_to_result_file.c_str(), "w");
  if (!fp) {
    std::cerr << "error: unable to open result file " << path_to_result_file << std::endl;
    return;
  }

  for (const Stats stat : stats)
  {
    fprintf(fp, "%s %f %f\n", stat.sequence_name_.c_str(), stat.t_err, stat.r_err);
    std::cout << stat.sequence_name_ << " " << std::fixed << std::setprecision(4) 
              << stat.t_err << " " << stat.r_err << std::endl;
  }

  if (!stats.empty()) {
    double t_err_sum = 0;
    double r_err_sum = 0;
    for (const auto& stat : stats) {
      t_err_sum += stat.t_err;
      r_err_sum += stat.r_err;
    }
    double t_err_avg = t_err_sum / stats.size();
    double r_err_avg = r_err_sum / stats.size();
    
    fprintf(fp, "Average: %f %f\n", t_err_avg, r_err_avg);
    std::cout << "Average: " << std::fixed << std::setprecision(4) 
              << t_err_avg << " " << r_err_avg << std::endl;
  }

  fclose(fp);
}

// Copyright 2019 ETH Zürich, Thomas Schöps
void WriteTrajectorySVG(
    std::ofstream& stream,
    int plot_size_in_pixels,
    const Eigen::Vector3f& min_vec,
    const Eigen::Vector3f& max_vec,
    const std::vector<Pose>& trajectory,
    const std::string& color,
    const float stroke_width,
    int dimension1,
    int dimension2) {
  
  std::ostringstream stroke_width_stream;
  stroke_width_stream << stroke_width;
  std::string stroke_width_string = stroke_width_stream.str();
  
  std::ostringstream half_stroke_width_stream;
  half_stroke_width_stream << (0.5 * stroke_width);
  std::string half_stroke_width_string = half_stroke_width_stream.str();
  
  bool within_polyline = false;
  
  for (std::size_t i = 0; i < trajectory.size() - 1; ++ i) {
    const Eigen::Vector3f& point = trajectory[i].transform.translation();
    Eigen::Vector3f plot_point = plot_size_in_pixels * (point - min_vec).cwiseQuotient(max_vec - min_vec);
    
    // Check if frames are consecutive (no gap)
    bool segment_valid = (trajectory[i+1].frame_id - trajectory[i].frame_id <= 5);  // Allow small gaps
    
    if (!segment_valid && !within_polyline) {
      stream << "<circle cx=\"" << plot_point.coeff(dimension1) << "\" cy=\"" << plot_point.coeff(dimension2) << "\" r=\"" << half_stroke_width_string << "\" fill=\"" << color << "\"/>\n";
      continue;
    }
    
    if (segment_valid && !within_polyline) {
      // Start new polyline
      stream << "<polyline points=\"";
      within_polyline = true;
    } else {
      // Write the space between two points
      stream << " ";
    }
    
    stream << plot_point.coeff(dimension1) << "," << plot_point.coeff(dimension2);
    
    if (!segment_valid && within_polyline) {
      // End polyline
      stream << "\" stroke=\"" << color << "\" stroke-width=\"" << stroke_width_string << "\" fill=\"none\" />\n";
      within_polyline = false;
    }
  }
  
  if (within_polyline) {
    // End polyline
    stream << "\" stroke=\"" << color << "\" stroke-width=\"" << stroke_width_string << "\" fill=\"none\" />\n";
  }
}

// Copyright 2019 ETH Zürich, Thomas Schöps
void PlotTrajectories(
    const std::string& path,
    int plot_size_in_pixels,
    const std::vector<Pose>& ground_truth,
    const std::vector<Pose>& aligned_estimate,
    int dimension1,
    int dimension2) {
  std::ofstream stream(path, std::ios::out);
  
  stream << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
  stream << "<svg width=\"" << plot_size_in_pixels << "\" height=\"" << plot_size_in_pixels
                          << "\" viewBox=\"0 0 " << plot_size_in_pixels << " " << plot_size_in_pixels
                          << "\" xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\">\n";
  
  // Determine plot extent based on the ground truth trajectory
  Eigen::Vector3f min_vec = Eigen::Vector3f::Constant(std::numeric_limits<double>::infinity());
  Eigen::Vector3f max_vec = Eigen::Vector3f::Constant(-1 * std::numeric_limits<double>::infinity());

  for (std::size_t i = 0; i < ground_truth.size(); ++ i) {
    min_vec = min_vec.cwiseMin(ground_truth[i].transform.translation());
    max_vec = max_vec.cwiseMax(ground_truth[i].transform.translation());
  }

  for (std::size_t i = 0; i < aligned_estimate.size(); ++ i) {
    min_vec = min_vec.cwiseMin(aligned_estimate[i].transform.translation());
    max_vec = max_vec.cwiseMax(aligned_estimate[i].transform.translation());
  }
  
  float largest_size = (max_vec - min_vec).maxCoeff();
  Eigen::Vector3f center = 0.5 * (min_vec + max_vec);
  constexpr float kSizeExtensionFactor = 1.1f;
  min_vec = center - 0.5 * Eigen::Vector3f::Constant(kSizeExtensionFactor * largest_size);
  max_vec = center + 0.5 * Eigen::Vector3f::Constant(kSizeExtensionFactor * largest_size);
  
  // Plot ground truth trajectory
  WriteTrajectorySVG(stream, plot_size_in_pixels, min_vec, max_vec, ground_truth, "green", 1, dimension1, dimension2);
  
  // Plot estimated trajectory
  WriteTrajectorySVG(stream, plot_size_in_pixels, min_vec, max_vec, aligned_estimate, "red", 1, dimension1, dimension2);
  
  stream << "</svg>\n";
  
  stream.close();
}

inline void dumpError(const std::string& path, const std::vector<ErrorPair>& errors, const std::vector<Pose>& poses_es, bool rpe=false) {
  std::ofstream file(path);

  if(rpe)
    file << "# frame_id rotation(deg/m) translation(%)" << std::endl;
  else
    file << "# frame_id rotation(deg) translation(m)" << std::endl;

  for (size_t i = 0; i < poses_es.size(); ++i) {
    file << poses_es[i].frame_id << " " << errors[i].r_err << " " << errors[i].t_err << std::endl;
  }

  file.close();
}

inline void eval(const std::string &path_to_gt, const std::string &path_to_es, bool plot, bool use_sim3 = true)
{
  std::string path_to_result = path_to_es + "/results";
  if (use_sim3) {
    path_to_result += "_sim3";
  } else {
    path_to_result += "_se3";
  }
  system(("mkdir -p " + path_to_result).c_str());

  std::vector<Stats> rpe_stats;
  std::vector<Stats> ate_stats;
  
  for (const std::string& sequence_name : SEQ_NAMES)
  {
    const std::string path_to_gt_file = path_to_gt + "/" + sequence_name + ".txt";
    const std::string path_to_es_file = path_to_es + "/" + sequence_name + ".txt";

    // Check if estimated file exists
    std::ifstream es_check(path_to_es_file);
    if (!es_check.good()) {
      std::cout << "Skipping sequence " << sequence_name << " (no estimated poses found)" << std::endl;
      continue;
    }
    es_check.close();

    std::vector<Pose> poses_gt = loadKittiGTPoses(path_to_gt_file);
    std::vector<Pose> poses_es_orig = loadEstimatedPoses(path_to_es_file);

    std::cout << "=============================================" << std::endl;
    std::cout << "processing: " << sequence_name << std::endl;
    std::cout << "GT poses: " << poses_gt.size() << std::endl;
    std::cout << "estimated poses: " << poses_es_orig.size() << std::endl;

    if (poses_gt.size() == 0 || poses_es_orig.size() == 0)
    {
      std::cout << "ERROR: could not read poses for sequence: " << sequence_name << std::endl;
      continue;
    }

    // Match poses by frame index
    const auto [gt_matched, es_matched] = matchByFrameIndex(poses_gt, poses_es_orig);
    
    std::cout << "matched poses: " << gt_matched.size() << std::endl << std::endl;

    if (gt_matched.size() == 0)
    {
      std::cout << "ERROR: no valid matched poses for: " << sequence_name << std::endl;
      continue;
    }

    const std::vector<Error> seq_err = computeSequenceErrors(gt_matched, es_matched);
    
    if (seq_err.empty()) {
      std::cout << "WARNING: no valid subsequences for RPE computation" << std::endl;
    } else {
      const auto [rpe_stat, RPE_errors] = computeSequenceRPE(seq_err, sequence_name, es_matched.size());
      rpe_stats.push_back(rpe_stat);
      
      if (plot) {
        dumpError(path_to_result + "/" + sequence_name + "_rpe.txt", RPE_errors, es_matched, true);
      }
    }

    const std::vector<Pose> poses_es_aligned = computeAlignedEstimate(gt_matched, es_matched, use_sim3);
    const auto [ate_stat, ATE_errors] = computeSequenceATE(gt_matched, poses_es_aligned, sequence_name);
    ate_stats.push_back(ate_stat);

    if (plot) {
      constexpr int kPlotSize = 600;
      PlotTrajectories(
          path_to_result + "/" + sequence_name + "_top.svg",
          kPlotSize,
          gt_matched,
          poses_es_aligned,
          0,
          2);  // X-Z plane (top view for KITTI)
      PlotTrajectories(
          path_to_result + "/" + sequence_name + "_front.svg",
          kPlotSize,
          gt_matched,
          poses_es_aligned,
          0,
          1);  // X-Y plane
      PlotTrajectories(
          path_to_result + "/" + sequence_name + "_side.svg",
          kPlotSize,
          gt_matched,
          poses_es_aligned,
          2,
          1);  // Z-Y plane
      dumpError(path_to_result + "/" + sequence_name + "_ate.txt", ATE_errors, es_matched);
    }
  }

  if (rpe_stats.size() > 0) {
    std::cout << std::endl << "RPE stats (sequence, t_err [%], r_err [deg/100m]):" << std::endl;
    computeAndPrintStats(rpe_stats, path_to_result + "/results_rpe.txt");
  }

  std::cout << std::endl << "ATE RMSE stats (sequence, t_err [m], r_err [deg]):" << std::endl;
  computeAndPrintStats(ate_stats, path_to_result + "/results_ate.txt");
}

int main(int argc, char *argv[])
{
  if (argc < 3)
  {
    std::cout << "KITTI Odometry Benchmark Evaluation Tool" << std::endl;
    std::cout << std::endl;
    std::cout << "usage: ./kitti_benchmark path_to_gt path_to_es [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Arguments:" << std::endl;
    std::cout << "  path_to_gt  : Path to KITTI GT poses directory (containing 00.txt, 01.txt, ...)" << std::endl;
    std::cout << "  path_to_es  : Path to estimated poses directory (containing 00.txt, 01.txt, ...)" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --plot      : Generate SVG trajectory plots and error dumps" << std::endl;
    std::cout << "  --se3       : Use SE(3) alignment instead of Sim(3) (for stereo/LiDAR)" << std::endl;
    std::cout << std::endl;
    std::cout << "GT format: 3x4 matrix per line (r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz)" << std::endl;
    std::cout << "ES format: timestamp tx ty tz qx qy qz qw" << std::endl;
    return 1;
  }

  const std::string &path_to_gt = argv[1];
  const std::string &path_to_es = argv[2];

  bool plot = false;
  bool use_sim3 = true;
  
  for (int i = 3; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--plot") {
      plot = true;
    } else if (arg == "--se3") {
      use_sim3 = false;
    }
  }

  std::cout << "KITTI Odometry Benchmark Evaluation" << std::endl;
  std::cout << "====================================" << std::endl;
  std::cout << "GT path: " << path_to_gt << std::endl;
  std::cout << "ES path: " << path_to_es << std::endl;
  std::cout << "Alignment: " << (use_sim3 ? "Sim(3)" : "SE(3)") << std::endl;
  std::cout << "Plot: " << (plot ? "enabled" : "disabled") << std::endl;
  std::cout << std::endl;

  eval(path_to_gt, path_to_es, plot, use_sim3);

  return 0;
}
