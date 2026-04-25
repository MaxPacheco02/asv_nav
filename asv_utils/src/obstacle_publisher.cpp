#include <algorithm>
#include <chrono>
#include <cstddef>
#include <eigen3/Eigen/Dense>
#include <functional>
#include <geometry_msgs/msg/detail/pose_stamped__struct.hpp>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>

#include "asv_interfaces/msg/obstacle.hpp"
#include "asv_interfaces/msg/obstacle_list.hpp"
#include "geometry_msgs/msg/pose2_d.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/color_rgba.hpp"
#include "std_msgs/msg/float64.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/transform_broadcaster.h>

struct MarkerProps {
  int type;
  double x, y, z, z_trans;
};

using namespace std::chrono_literals;

class ObstaclePublisher : public rclcpp::Node {
public:
  ObstaclePublisher() : Node("obstacle_publisher") {
    near_obs_pub_ = this->create_publisher<asv_interfaces::msg::ObstacleList>(
        "/mpc/near_obs", 10);

    obs_pub_ = this->create_publisher<asv_interfaces::msg::ObstacleList>(
        "/mpc/obs", 10);

    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/mpc/obs_marker_arr", 10);

    near_marker_pub_ =
        this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/mpc/obs_marker_arr_near", 10);

    pose_sub_ = this->create_subscription<geometry_msgs::msg::Pose2D>(
        "/asv/state/pose", 10, [this](const geometry_msgs::msg::Pose2D &msg) {
          pose_ << msg.x, msg.y;
        });

    dyn_obs_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/rviz/dyn_obs", 1,
        [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
          auto &q = msg->pose.orientation;
          double ang = std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                                  1.0 - 2.0 * (q.y * q.y + q.z * q.z));

          dyn_obs[dyn_idx][0] = msg->pose.position.x;
          dyn_obs[dyn_idx][1] = msg->pose.position.y;
          dyn_obs[dyn_idx][2] = max_vel * cos(ang);
          dyn_obs[dyn_idx][3] = max_vel * sin(ang);
          dyn_idx = (dyn_idx + 1) % dyn_obs_n;
        });

    timer_ = this->create_wall_timer(
        100ms, std::bind(&ObstaclePublisher::timer_callback, this));

    dummy_obs.color = 5;
    dummy_obs.type = "NaN";

    // Initialize random dynamic obstacles
    // Seed RNG
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist_x(area[1],
                                                  area[0]); // -50 to 1500
    std::uniform_real_distribution<double> dist_y(area[2],
                                                  area[3]); // -200 to 200
    std::uniform_real_distribution<double> dist_vx(-max_vel, max_vel);
    std::uniform_real_distribution<double> dist_vy(-max_vel, max_vel);

    large_scale =
        geometry_msgs::build<geometry_msgs::msg::Vector3>().x(50).y(50).z(50);
    for (int i = 0; i < dyn_obs_n; i++) {
      dyn_obs[i][0] = dist_x(rng);
      dyn_obs[i][1] = dist_y(rng);
      dyn_obs[i][2] = dist_vx(rng);
      dyn_obs[i][3] = dist_vy(rng);
    }

    for (int i = 0; i < dyn_obs_n; i++) {

      double x = dyn_obs[i][0];
      double y = dyn_obs[i][1];
      double v_x = dyn_obs[i][2];
      double v_y = dyn_obs[i][3];

      dyn_obs_id[i] = obs_.obs_list.size();
      obs_.obs_list.push_back(build_obs(x, y, v_x, v_y));
      marker_arr.markers.push_back(
          build_marker(marker_arr.markers.size(), x, y, std::atan2(v_y, v_x)));
    }
  }

private:
  rclcpp::TimerBase::SharedPtr timer_;

  rclcpp::Publisher<asv_interfaces::msg::ObstacleList>::SharedPtr near_obs_pub_,
      obs_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      near_marker_pub_,
      marker_pub_;

  rclcpp::Subscription<asv_interfaces::msg::ObstacleList>::SharedPtr obs_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Pose2D>::SharedPtr pose_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr dyn_obs_sub_;

  asv_interfaces::msg::ObstacleList near_obs_, obs_;
  visualization_msgs::msg::MarkerArray near_marker_arr, marker_arr;
  Eigen::Vector2d pose_;
  asv_interfaces::msg::Obstacle dummy_obs{};

  geometry_msgs::msg::Vector3 large_scale;

  int64_t color = 3;
  std::string type = "marker";
  std::string uuid = "";

  double max_vel{10.0};
  static const int dyn_obs_n{3};
  double area[4]{-1000, 4500, -1400, 1400};
  double dyn_obs[dyn_obs_n][4]{
      {100., 2., -10.20, 1.0},
      {100., 0., 3.0, 4.0},
      {0., -50., 3.5, 2.0},
  };
  int dyn_idx{0};
  int dyn_obs_id[dyn_obs_n];

  int color_list[6][4]{{1, 0, 0, 1}, {0, 1, 0, 1}, {0, 0, 1, 1},
                       {1, 1, 0, 1}, {0, 0, 0, 1}, {0, 0, 0, 0}};
  std::map<std::string, MarkerProps> marker_type = {
      {"round", MarkerProps{2, 0.5, 0.5, 0.5, 0}},
      {"boat", MarkerProps{2, 1.0, 1.0, 1.0, 0}},
      {"marker", MarkerProps{0, 20.0, 5.0, 5.0, 0.25}},
      {"picture", MarkerProps{1, 0.5, 0.5, 0.5, 0.25}},
  };

  asv_interfaces::msg::Obstacle build_obs(double x, double y, double v_x,
                                          double v_y) {
    return asv_interfaces::build<asv_interfaces::msg::Obstacle>()
        .x(x)
        .y(y)
        .v_x(v_x)
        .v_y(v_y)
        .color(color)
        .type(type)
        .uuid(uuid);
  }

  visualization_msgs::msg::Marker build_marker(int id, double x, double y,
                                               double ang) {
    visualization_msgs::msg::Marker marker;

    int r{color_list[color][0]}, g{color_list[color][1]},
        b{color_list[color][2]}, a{color_list[color][3]};

    tf2::Quaternion q;
    q.setRPY(0, 0, ang);

    marker.header.frame_id = "world";
    marker.color =
        std_msgs::build<std_msgs::msg::ColorRGBA>().r(r).g(g).b(b).a(a);
    marker.action = 0;
    marker.id = id;
    marker.type = marker_type[type].type;
    marker.scale = geometry_msgs::build<geometry_msgs::msg::Vector3>()
                       .x(marker_type[type].x)
                       .y(marker_type[type].y)
                       .z(marker_type[type].z);
    marker.pose.position.x = x;
    marker.pose.position.y = y;
    marker.pose.position.z = marker_type[type].z_trans;
    marker.pose.orientation = tf2::toMsg(q);
    return marker;
  }

  void timer_callback() {
    update_dyn();

    if (obs_.obs_list.empty())
      return;

    std::vector<std::pair<double, int>> obs_dist_v;
    for (size_t i = 0; i < obs_.obs_list.size(); i++) {
      obs_dist_v.push_back(
          std::pair<double, int>{obs_dist(obs_.obs_list[i], pose_), i});
    }
    std::sort(obs_dist_v.begin(), obs_dist_v.end());

    near_obs_.obs_list.clear();
    near_marker_arr.markers.clear();
    size_t i = 0;
    while (near_obs_.obs_list.size() < 3) {
      if (i < obs_dist_v.size()) {
        int idx = obs_dist_v[i].second;
        near_obs_.obs_list.push_back(obs_.obs_list[idx]);
        near_marker_arr.markers.push_back(marker_arr.markers[idx]);
        near_marker_arr.markers[i].id = i;
        near_marker_arr.markers[i].color.a = 0.5;
        near_marker_arr.markers[i].scale = large_scale;
        near_marker_arr.markers[i].type = 2;
      } else {
        near_obs_.obs_list.push_back(dummy_obs);
      }
      i++;
    }

    obs_pub_->publish(obs_);
    near_obs_pub_->publish(near_obs_);
    marker_pub_->publish(marker_arr);
    near_marker_pub_->publish(near_marker_arr);
  }

  void update_dyn() {
    double dt = 0.1;
    for (int i = 0; i < dyn_obs_n; i++) {
      dyn_obs[i][0] += dyn_obs[i][2] * dt;
      dyn_obs[i][1] += dyn_obs[i][3] * dt;

      if (dyn_obs[i][0] < area[0]) {
        dyn_obs[i][2] = fabs(dyn_obs[i][2]);
      } else if (dyn_obs[i][0] > area[1]) {
        dyn_obs[i][2] = -fabs(dyn_obs[i][2]);
      }
      if (dyn_obs[i][1] < area[2]) {
        dyn_obs[i][3] = fabs(dyn_obs[i][3]);
      } else if (dyn_obs[i][1] > area[3]) {
        dyn_obs[i][3] = -fabs(dyn_obs[i][3]);
      }

      obs_.obs_list[dyn_obs_id[i]].x = dyn_obs[i][0];
      obs_.obs_list[dyn_obs_id[i]].y = dyn_obs[i][1];
      obs_.obs_list[dyn_obs_id[i]].v_x = dyn_obs[i][2];
      obs_.obs_list[dyn_obs_id[i]].v_y = dyn_obs[i][3];
      marker_arr.markers[dyn_obs_id[i]].pose.position.x = dyn_obs[i][0];
      marker_arr.markers[dyn_obs_id[i]].pose.position.y = dyn_obs[i][1];
      tf2::Quaternion q;
      q.setRPY(0, 0, std::atan2(dyn_obs[i][3], dyn_obs[i][2]));
      marker_arr.markers[dyn_obs_id[i]].pose.orientation = tf2::toMsg(q);
    }
  }

  double obs_dist(asv_interfaces::msg::Obstacle obs, const Eigen::Vector2d &p) {
    Eigen::Vector2d obs_{obs.x, obs.y};
    return (obs_ - p).norm();
  }
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ObstaclePublisher>());
  rclcpp::shutdown();
  return 0;
}
