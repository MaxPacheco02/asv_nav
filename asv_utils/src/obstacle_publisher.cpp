#include <algorithm>
#include <chrono>
#include <cstddef>
#include <eigen3/Eigen/Dense>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>

#include "asv_interfaces/msg/obstacle.hpp"
#include "asv_interfaces/msg/obstacle_list.hpp"
#include "geometry_msgs/msg/pose2_d.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/color_rgba.hpp"
#include "std_msgs/msg/float64.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

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
        "/marker_array", 10);

    pose_sub_ = this->create_subscription<geometry_msgs::msg::Pose2D>(
        "/asv/state/pose", 10, [this](const geometry_msgs::msg::Pose2D &msg) {
          pose_ << msg.x, msg.y;
        });

    timer_ = this->create_wall_timer(
        100ms, std::bind(&ObstaclePublisher::timer_callback, this));

    dummy_obs.color = 5;
    dummy_obs.type = "NaN";

    for (int i = 0; i < dyn_obs_n; i++) {
      asv_interfaces::msg::Obstacle obs;
      visualization_msgs::msg::Marker marker;

      double x = dyn_obs[i][0];
      double y = dyn_obs[i][1];
      double v_x = dyn_obs[i][2];
      double v_y = dyn_obs[i][3];
      int64_t color = 3;
      std::string type = "marker";
      std::string uuid = "";
      obs = asv_interfaces::build<asv_interfaces::msg::Obstacle>()
                .x(x)
                .y(y)
                .v_x(v_x)
                .v_y(v_y)
                .color(color)
                .type(type)
                .uuid(uuid);
      dyn_obs_id[i] = obs_.obs_list.size();
      obs_.obs_list.push_back(obs);

      int r{color_list[color][0]}, g{color_list[color][1]},
          b{color_list[color][2]}, a{color_list[color][3]};

      marker.header.frame_id = "world";
      marker.color =
          std_msgs::build<std_msgs::msg::ColorRGBA>().r(r).g(g).b(b).a(a);
      marker.action = 0;
      marker.id = marker_arr.markers.size();
      marker.type = marker_type[type].type;
      marker.scale = geometry_msgs::build<geometry_msgs::msg::Vector3>()
                         .x(marker_type[type].x)
                         .y(marker_type[type].y)
                         .z(marker_type[type].z);
      marker.pose.position.x = x;
      marker.pose.position.y = y;
      marker.pose.position.z = marker_type[type].z_trans;
      marker_arr.markers.push_back(marker);
    }
  }

private:
  rclcpp::TimerBase::SharedPtr timer_;

  rclcpp::Publisher<asv_interfaces::msg::ObstacleList>::SharedPtr near_obs_pub_,
      obs_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      marker_pub_;

  rclcpp::Subscription<asv_interfaces::msg::ObstacleList>::SharedPtr obs_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Pose2D>::SharedPtr pose_sub_;

  asv_interfaces::msg::ObstacleList near_obs_, obs_;
  visualization_msgs::msg::MarkerArray marker_arr;
  Eigen::Vector2d pose_;
  asv_interfaces::msg::Obstacle dummy_obs{};

  static const int dyn_obs_n{3};
  double area[4]{500, -50, -200, 200};
  double dyn_obs[dyn_obs_n][4]{
      // {100., 2., -10.20, 1.0},
      // {100., 0., 3.0, 4.0},
      // {0., -50., 3.5, 2.0},
      // {500., 20., 2.0, 2.8},
      // {500., 0., -1.0, -2.8},
      // {200., 80., 3.0, -2.8},
      // {300., -30., 5.0, -1.23},
      // {0., 80., 2.0, -1.0},
      // TMP
      {400., -20, 0.0, 0.0},
      {165., 35., 0.0, 0.0},
      {600., -80., 0.0, 0.0},
      // {500., 20., 0.0, 0.0},  {500., 0., -0.0, 0.0}, {200., 80., 0.0, 0.0},
      // {300., -30., 0.0, 0.0}, {0., 80., 0.0, 0.0},
  };
  int dyn_obs_id[dyn_obs_n];

  int color_list[6][4]{{1, 0, 0, 1}, {0, 1, 0, 1}, {0, 0, 1, 1},
                       {1, 1, 0, 1}, {0, 0, 0, 1}, {0, 0, 0, 0}};
  std::map<std::string, MarkerProps> marker_type = {
      {"round", MarkerProps{2, 0.5, 0.5, 0.5, 0}},
      {"boat", MarkerProps{2, 1.0, 1.0, 1.0, 0}},
      {"marker", MarkerProps{3, 5.0, 5.0, 1.0, 0.25}},
      {"picture", MarkerProps{1, 0.5, 0.5, 0.5, 0.25}},
  };

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
    size_t i = 0;
    while (near_obs_.obs_list.size() < 3) {
      if (i < obs_dist_v.size())
        near_obs_.obs_list.push_back(obs_.obs_list[obs_dist_v[i].second]);
      else
        near_obs_.obs_list.push_back(dummy_obs);
      i++;
    }

    obs_pub_->publish(obs_);
    near_obs_pub_->publish(near_obs_);
    marker_pub_->publish(marker_arr);
  }

  void update_dyn() {
    double dt = 0.01;
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
