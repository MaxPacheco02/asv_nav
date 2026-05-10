#include "rclcpp/rclcpp.hpp"
#include <eigen3/Eigen/Dense>
#include <limits>
#include <vector>

#include "asv_interfaces/msg/spline.hpp"
#include "asv_interfaces/msg/spline_params.hpp"

#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/vector3.hpp"

#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"

#include "std_msgs/msg/color_rgba.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include "asv_control/utils/CatmulRom.h"

using namespace std::chrono_literals;

class SplinePublisherNode : public rclcpp::Node {
public:
  SplinePublisherNode() : Node("spline_publisher_node") {
    using namespace std::placeholders;

    // Declare and read the closed-loop parameter
    this->declare_parameter<bool>("closed", false);
    closed_ = this->get_parameter("closed").as_bool();

    this->declare_parameter<double>("marker_scale", 1.0);
    marker_scale_ = this->get_parameter("marker_scale").as_double();
    this->declare_parameter<double>("lookahead", 300.0);
    lookahead = this->get_parameter("lookahead").as_double();

    this->declare_parameter<std::vector<double>>("waypoints",
                                                 std::vector<double>{});
    auto wp_flat = this->get_parameter("waypoints").as_double_array();
    if (wp_flat.size() >= 4 && wp_flat.size() % 2 == 0) {
      ref.clear();
      for (size_t i = 0; i + 1 < wp_flat.size(); i += 2)
        ref.push_back({wp_flat[i], wp_flat[i + 1]});
    }

    spline_path_pub_ =
        this->create_publisher<nav_msgs::msg::Path>("/asv/path_ref", 10);
    dummy_path_pub_ =
        this->create_publisher<nav_msgs::msg::Path>("/asv/dummy_path_ref", 10);
    s_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
        "/spline_marker", 10);
    la_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
        "/lookahead_marker", 10);
    spline_params_pub_ =
        this->create_publisher<asv_interfaces::msg::SplineParams>(
            "/mpc/spline_params", 10);
    waypoint_labels_pub_ =
        this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/waypoint_labels", 10);
    waypoint_debug_pub_ =
        this->create_publisher<std_msgs::msg::Float64MultiArray>(
            "/debug/spline_wps", 10);

    // Goal from mission_handler node
    mission_goal_sub_ =
        this->create_subscription<geometry_msgs::msg::PoseArray>(
            "/asv/goals/pose_array", 1,
            [this](const geometry_msgs::msg::PoseArray::SharedPtr msg) {
              if (msg->poses.empty())
                return;

              // Only replan if the final destination actually changed
              auto &last = msg->poses.back();
              if (!ref.empty()) {
                double dx = last.position.x - last_goal_x_;
                double dy = last.position.y - last_goal_y_;
                if (std::sqrt(dx * dx + dy * dy) < 0.1) {
                  // same destination — just append new intermediate points
                  // without resetting closest_idx
                  ref.clear();
                  for (size_t i = 0; i < msg->poses.size(); i++)
                    ref.push_back(trans(p_to_v(msg->poses[i]), 0.0));
                  if (!closed_) {
                    ref.push_back(trans(p_to_v(msg->poses.back()), dist));
                    dummy_ref_ = trans(p_to_v(msg->poses.back()), 2 * dist);
                  }
                  // don't reset closest_idx or lap_ here
                  update_spline_params();
                  return;
                }
              }

              // Final destination changed — full replan
              last_goal_x_ = last.position.x;
              last_goal_y_ = last.position.y;
              ref.clear();
              if (!closed_)
                ref.push_back(trans(p_to_v(msg->poses[0]), -dist));
              for (size_t i = 0; i < msg->poses.size(); i++)
                ref.push_back(trans(p_to_v(msg->poses[i]), 0.0));
              if (!closed_) {
                ref.push_back(trans(p_to_v(msg->poses.back()), dist));
                dummy_ref_ = trans(p_to_v(msg->poses.back()), 2 * dist);
              }
              closest_idx = 0;
              last_idx = -1;
              lap_ = 0;
              update_spline_params();
            });

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/asv/state/odom", 1,
        [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
          auto &q = msg->pose.pose.orientation;

          asv.x() = msg->pose.pose.position.x;
          asv.y() = msg->pose.pose.position.y;
          asv.z() = std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                               1.0 - 2.0 * (q.y * q.y + q.z * q.z));
        });

    // Goal as a PoseStamped msg (for RViz)
    goal_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/goal_pose", 1,
        [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
          ref.clear();
          if (!closed_) {
            ref.push_back(trans(asv, -dist));
          }
          ref.push_back(trans(asv, 0.0));

          auto &q = msg->pose.orientation;
          tmp.x() = msg->pose.position.x;
          tmp.y() = msg->pose.position.y;
          tmp.z() = std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                               1.0 - 2.0 * (q.y * q.y + q.z * q.z));
          ref.push_back(trans(tmp, 0.0));
          if (!closed_) {
            ref.push_back(trans(tmp, dist));
            dummy_ref_ = trans(tmp, 2 * dist);
          }

          lap_ = 0;
          closest_idx = 0;
          last_idx = -1;
          update_spline_params();
        });

    goal_pose_to_sub_ =
        this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/goal_to", 1,
            [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
              auto &q = msg->pose.orientation;
              tmp.x() = msg->pose.position.x;
              tmp.y() = msg->pose.position.y;
              tmp.z() = std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                                   1.0 - 2.0 * (q.y * q.y + q.z * q.z));
              if (closed_) {
                ref.push_back(trans(tmp, 0.0));
              } else {
                ref[ref.size() - 1] = trans(tmp, 0.0);
                ref.push_back(trans(tmp, dist));
                dummy_ref_ = trans(tmp, 2 * dist);
              }

              update_spline_params();
            });

    goal_pose_from_sub_ =
        this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/goal_from", 1,
            [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
              ref.clear();
              auto &q = msg->pose.orientation;
              tmp.x() = msg->pose.position.x;
              tmp.y() = msg->pose.position.y;
              tmp.z() = std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                                   1.0 - 2.0 * (q.y * q.y + q.z * q.z));
              if (!closed_)
                ref.push_back(trans(tmp, -dist));
              ref.push_back(trans(tmp, 0.0));
              if (!closed_)
                ref.push_back(trans(tmp, dist));

              lap_ = 0;
              closest_idx = 0;
              last_idx = -1;
              update_spline_params();
            });

    timer_ = this->create_wall_timer(
        100ms, std::bind(&SplinePublisherNode::update, this));

    path_msg.header.frame_id = "world";

    // Setup spline marker
    s_marker_msg.id = 0;
    s_marker_msg.type = visualization_msgs::msg::Marker::SPHERE;
    s_marker_msg.action = 0;
    s_marker_msg.scale = geometry_msgs::build<geometry_msgs::msg::Vector3>()
                             .x(5.0 * marker_scale_)
                             .y(5.0 * marker_scale_)
                             .z(5.0 * marker_scale_);
    s_marker_msg.color =
        std_msgs::build<std_msgs::msg::ColorRGBA>().r(0).g(0).b(1).a(1);

    // Setup lookahead marker
    la_marker_msg.id = 0;
    la_marker_msg.type = visualization_msgs::msg::Marker::SPHERE;
    la_marker_msg.action = 0;
    la_marker_msg.scale = geometry_msgs::build<geometry_msgs::msg::Vector3>()
                              .x(5.0 * marker_scale_)
                              .y(5.0 * marker_scale_)
                              .z(5.0 * marker_scale_);
    la_marker_msg.color =
        std_msgs::build<std_msgs::msg::ColorRGBA>().r(1).g(0).b(0).a(1);

    update_spline_params();
  }

protected:
  void update() {
    path_msg.poses.clear();
    path_msg.header.stamp = this->get_clock()->now();
    dummy_path_msg.header = path_msg.header;
    geometry_msgs::msg::PoseStamped tmp_pose;
    tmp_pose.header = path_msg.header;
    s_marker_msg.header = path_msg.header;
    la_marker_msg.header = path_msg.header;

    if (s_.size() > 0) {
      Eigen::Vector2d tmp_v;
      Eigen::Vector2d closest_p_tmp, closest_p;
      double closest_t = 0.0, closest_t_tmp;
      double closest_dist = std::numeric_limits<double>::max();
      int N = static_cast<int>(s_.size());

      for (int i = 0; i < N; i++) {

        for (double t = 0; t <= 1; t += 1.0 / (n_ - 1)) {
          tmp_v = s_[i].get_s(t);
          tmp_pose.pose.position.x = tmp_v.x();
          tmp_pose.pose.position.y = tmp_v.y();
          tmp_pose.pose.position.z = 0;
          path_msg.poses.push_back(tmp_pose);
        }

        closest_t_tmp = s_[i].closest_t(asv);
        closest_p_tmp = s_[i].get_s(closest_t_tmp);

        // Neighbor check: in closed mode, allow wraparound (N-1 ↔ 0)
        int diff = std::abs(i - closest_idx);
        int neighbor_dist = closed_ ? std::min(diff, N - diff) : diff;

        if (distance(asv, closest_p_tmp) < closest_dist && neighbor_dist <= 1) {
          closest_t = closest_t_tmp;
          closest_dist = distance(asv, closest_p_tmp);
          closest_idx = i;
        }
      }

      // Detect lap transitions across the seam (only in closed mode)
      if (closed_ && last_idx >= 0) {
        if (last_idx == N - 1 && closest_idx == 0) {
          lap_++;
        } else if (last_idx == 0 && closest_idx == N - 1) {
          lap_--; // backwards across the seam
        }
      }

      closest_p = s_[closest_idx].get_s(closest_t);

      // For length L, we want to find a t+dt such that s(t+dt) is at [dist]
      // from s(t). To map L to dist: L is to 1, what dist is to dt -> dt =
      // dist/L
      L_ = s_[closest_idx].L_;
      double la_t = s_[closest_idx].get_la(closest_t, lookahead);
      Eigen::Vector2d la_p = s_[closest_idx].get_s(la_t);

      bool la_wrapped = false;
      int la_next_idx = closest_idx;
      double la_frac = la_t;

      if (la_t == 1.0 && (closed_ || closest_idx + 1 < N)) {
        // la_t is most likely saturated and there still are splines left to
        // cover (or we wrap around in closed mode)
        la_next_idx = closed_ ? (closest_idx + 1) % N : closest_idx + 1;
        double rem_dist =
            lookahead - s_[closest_idx].get_arc_length(closest_t, la_t);
        double next_la_t = s_[la_next_idx].get_la(0.0, rem_dist);
        la_t = 1 + next_la_t;
        la_p = s_[la_next_idx].get_s(next_la_t);
        la_frac = next_la_t;
        la_wrapped = true;
      }

      s_marker_msg.pose.position.x = closest_p.x();
      s_marker_msg.pose.position.y = closest_p.y();

      la_marker_msg.pose.position.x = la_p.x();
      la_marker_msg.pose.position.y = la_p.y();

      // Emit monotonically-increasing t in closed mode
      if (closed_) {
        spline_params_msg.t = lap_ * N + closest_idx + closest_t;

        if (la_wrapped) {
          // If the lookahead crossed the seam (i.e. wrapped from N-1 to 0),
          // it belongs to the next lap
          int la_lap = lap_ + ((la_next_idx < closest_idx) ? 1 : 0);
          spline_params_msg.t_la = la_lap * N + la_next_idx + la_frac;
        } else {
          spline_params_msg.t_la = lap_ * N + closest_idx + la_t;
        }
      } else {
        spline_params_msg.t = closest_idx + closest_t;
        spline_params_msg.t_la = closest_idx + la_t;
      }

      if (last_idx != closest_idx) {
        spline_params_msg.x = to_spline_msg(s_[closest_idx].s_, 0);
        spline_params_msg.y = to_spline_msg(s_[closest_idx].s_, 1);

        // Next-spline lookup with wraparound in closed mode
        if (closed_) {
          int next_idx = (closest_idx + 1) % N;
          spline_params_msg.x_next = to_spline_msg(s_[next_idx].s_, 0);
          spline_params_msg.y_next = to_spline_msg(s_[next_idx].s_, 1);
        } else if (closest_idx + 1 < N) {
          spline_params_msg.x_next = to_spline_msg(s_[closest_idx + 1].s_, 0);
          spline_params_msg.y_next = to_spline_msg(s_[closest_idx + 1].s_, 1);
        } else {
          spline_params_msg.x_next = to_spline_msg(dummy_s_.s_, 0);
          spline_params_msg.y_next = to_spline_msg(dummy_s_.s_, 1);
        }
      }

      last_idx = closest_idx;

      spline_params_msg.length = L_;

      // at_last_segment is meaningless in closed mode
      spline_params_msg.at_last_segment =
          closed_ ? false : (closest_idx >= N - 1);
    }

    // These saturation guards only apply in open mode — in closed mode, t can
    // legitimately exceed s_.size() because of the lap counter
    if (!closed_) {
      if (spline_params_msg.t_la == s_.size()) {
        spline_params_msg.t_la -= 0.001;
      }
      if (spline_params_msg.t == s_.size()) {
        spline_params_msg.t -= 0.001;
      }
    }

    visualization_msgs::msg::MarkerArray label_array;
    for (size_t i = 0; i < ref.size(); i++) {
      visualization_msgs::msg::Marker m;
      m.header = path_msg.header;
      m.ns = "waypoints";
      m.id = static_cast<int>(i);
      m.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.pose.position.x = ref[i].x();
      m.pose.position.y = ref[i].y();
      m.pose.position.z = 2.0;
      m.pose.orientation.w = 1.0;
      m.scale.z = 5.0 * marker_scale_;
      m.color = std_msgs::build<std_msgs::msg::ColorRGBA>().r(1).g(1).b(0).a(1);
      m.text = std::to_string(i);
      m.lifetime = rclcpp::Duration::from_seconds(0.15);
      label_array.markers.push_back(m);
    }

    std_msgs::msg::Float64MultiArray wp_debug_msg;
    for (const auto &wp : ref) {
      wp_debug_msg.data.push_back(wp.x());
      wp_debug_msg.data.push_back(wp.y());
    }

    spline_path_pub_->publish(path_msg);
    dummy_path_pub_->publish(dummy_path_msg);
    s_marker_pub_->publish(s_marker_msg);
    la_marker_pub_->publish(la_marker_msg);
    spline_params_pub_->publish(spline_params_msg);
    waypoint_labels_pub_->publish(label_array);
    waypoint_debug_pub_->publish(wp_debug_msg);
  }

  void update_spline_params() {
    s_.clear();

    if (closed_) {
      if (ref.size() < 3)
        return;
      size_t N = ref.size();
      s_.resize(N); // one segment per waypoint, looping back to the start
      for (size_t i = 0; i < N; i++) {
        s_[i].update(ref[(i + N - 1) % N], ref[i], ref[(i + 1) % N],
                     ref[(i + 2) % N]);
      }
      // dummy_s_ isn't really meaningful in closed mode, but keep it valid
      // (it gets used as the visual "preview" path)
      dummy_s_.update(ref[N - 1], ref[0], ref[1], ref[2]);
    } else {
      if (ref.size() < 4)
        return;
      s_.resize(ref.size() - 3);
      for (size_t i = 0; i < s_.size(); i++) {
        s_[i].update(ref[i], ref[i + 1], ref[i + 2], ref[i + 3]);
      }
      int i = ref.size() - 3;
      dummy_s_.update(ref[i], ref[i + 1], ref[i + 2], dummy_ref_);
    }

    update_dummy_msg();
  }

  void update_dummy_msg() {
    dummy_path_msg.poses.clear();
    Eigen::Vector2d tmp_v;
    geometry_msgs::msg::PoseStamped tmp_pose;
    for (double t = 0; t <= 1; t += 1.0 / (n_ - 1)) {
      tmp_v = dummy_s_.get_s(t);
      tmp_pose.pose.position.x = tmp_v.x();
      tmp_pose.pose.position.y = tmp_v.y();
      dummy_path_msg.poses.push_back(tmp_pose);
    }
  }

private:
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr spline_path_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr dummy_path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr s_marker_pub_,
      la_marker_pub_;
  rclcpp::Publisher<asv_interfaces::msg::SplineParams>::SharedPtr
      spline_params_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      waypoint_labels_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr
      waypoint_debug_pub_;

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr
      goal_pose_from_sub_,
      goal_pose_to_sub_, goal_pose_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr
      mission_goal_sub_;

  nav_msgs::msg::Path path_msg;
  nav_msgs::msg::Path dummy_path_msg;
  visualization_msgs::msg::Marker s_marker_msg, la_marker_msg;
  asv_interfaces::msg::SplineParams spline_params_msg;

  rclcpp::TimerBase::SharedPtr timer_;

  std::vector<CatmulRom> s_;
  CatmulRom dummy_s_;
  Eigen::Vector2d dummy_ref_{-0.5, -1};
  double L_{0.0};
  int n_{20};
  double dist{0.1};
  double marker_scale_{1.0};
  int closest_idx{-1};
  int last_idx{-1};
  int lap_{0};
  bool closed_{false};
  double lookahead = 300.0;

  std::vector<Eigen::Vector2d> ref{{-10, 0},     {-5, 0},     {500, 200},
                                   {1300, -200}, {1900, 200}, {2500, -200}};
  Eigen::Vector3d asv, tmp;

  double last_goal_x_{std::numeric_limits<double>::quiet_NaN()};
  double last_goal_y_{std::numeric_limits<double>::quiet_NaN()};

  Eigen::Vector3d p_to_v(geometry_msgs::msg::Pose p) {
    Eigen::Vector3d v;
    auto &q = p.orientation;
    v.x() = p.position.x;
    v.y() = p.position.y;
    v.z() = std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                       1.0 - 2.0 * (q.y * q.y + q.z * q.z));
    return v;
  }

  Eigen::Vector2d trans(Eigen::Vector3d v, double dist) {
    Eigen::Vector2d w, p;
    w << v(0), v(1);
    p << std::cos(v(2)), std::sin(v(2));
    return w + dist * p;
  }

  double distance(Eigen::Vector3d a, Eigen::Vector2d b) {
    Eigen::Vector2d c{a.x(), a.y()};
    return (b - c).norm();
  }

  asv_interfaces::msg::Spline to_spline_msg(Segment s, int i) {
    return asv_interfaces::build<asv_interfaces::msg::Spline>()
        .a(s.a[i])
        .b(s.b[i])
        .c(s.c[i])
        .d(s.d[i]);
  }
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SplinePublisherNode>());
  rclcpp::shutdown();
  return 0;
}
