// Mirrors the obstacles on /mpc/obs into gz sim as static, visual-only models.
// They are not simulated: every ObstacleList they get teleported to where the
// obstacle_publisher says, so the gz recording shows the same scenario the
// MPC is avoiding.

#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/entity_factory.pb.h>
#include <gz/msgs/pose_v.pb.h>
#include <gz/transport/Node.hh>

#include "asv_interfaces/msg/obstacle_list.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "rclcpp/rclcpp.hpp"

class GzObstacleNode : public rclcpp::Node {
public:
  GzObstacleNode() : Node("gz_obstacle_node") {
    world_ = this->declare_parameter<std::string>("world", "waves");
    prefix_ = this->declare_parameter<std::string>("model_prefix", "obs_");
    mesh_ = this->declare_parameter<std::string>("mesh",
                                                 "file://duck/meshes/duck.dae");
    mesh_scale_ = this->declare_parameter<double>("mesh_scale", 0.5);
    // Offset from the model origin to the mesh origin, [x, y, z, roll]
    mesh_offset_ = this->declare_parameter<std::vector<double>>(
        "mesh_offset", std::vector<double>{0.0, 0.0, 0.1, 1.5708});
    if (mesh_offset_.size() != 4) {
      RCLCPP_WARN(this->get_logger(),
                  "mesh_offset needs 4 values [x, y, z, roll]. Using zeros");
      mesh_offset_ = {0.0, 0.0, 0.0, 0.0};
    }
    z_ = this->declare_parameter<double>("z", 0.0);
    // Below this speed the heading is kept instead of atan2(v_y, v_x)
    min_heading_vel_ = this->declare_parameter<double>("min_heading_vel", 0.05);

    // ROS world = gz world - zero (see odom_converter_node)
    zero_sub_ = this->create_subscription<geometry_msgs::msg::Point>(
        "/usv/state/zero", rclcpp::QoS(1).transient_local(),
        [this](const geometry_msgs::msg::Point &msg) {
          zero_x_ = msg.x;
          zero_y_ = msg.y;
          has_zero_ = true;
        });

    obs_sub_ = this->create_subscription<asv_interfaces::msg::ObstacleList>(
        "/mpc/obs", 10,
        [this](const asv_interfaces::msg::ObstacleList &msg) { obs_cb(msg); });
  }

private:
  rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr zero_sub_;
  rclcpp::Subscription<asv_interfaces::msg::ObstacleList>::SharedPtr obs_sub_;

  gz::transport::Node gz_node_;

  std::string world_, prefix_, mesh_;
  double mesh_scale_{1.0};
  std::vector<double> mesh_offset_;
  double z_{0.0};
  double min_heading_vel_{0.05};

  bool has_zero_{false};
  double zero_x_{0.0}, zero_y_{0.0};

  // Number of obstacle models requested so far, and their last headings
  size_t n_spawned_{0};
  std::vector<double> yaw_;

  void obs_cb(const asv_interfaces::msg::ObstacleList &msg) {
    if (!has_zero_) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                           "Waiting for /usv/state/zero");
      return;
    }

    for (size_t i = yaw_.size(); i < msg.obs_list.size(); i++) {
      yaw_.push_back(0.0);
    }

    gz::msgs::Pose_V poses;
    for (size_t i = 0; i < msg.obs_list.size(); i++) {
      const auto &obs = msg.obs_list[i];
      if (std::hypot(obs.v_x, obs.v_y) > min_heading_vel_) {
        yaw_[i] = std::atan2(obs.v_y, obs.v_x);
      }
      double x = obs.x + zero_x_;
      double y = obs.y + zero_y_;

      if (i >= n_spawned_) {
        spawn(i, x, y);
        n_spawned_++;
        continue;
      }

      auto *p = poses.add_pose();
      p->set_name(model_name(i));
      p->mutable_position()->set_x(x);
      p->mutable_position()->set_y(y);
      p->mutable_position()->set_z(z_);
      p->mutable_orientation()->set_w(std::cos(yaw_[i] / 2.0));
      p->mutable_orientation()->set_z(std::sin(yaw_[i] / 2.0));
    }

    if (poses.pose_size() > 0) {
      gz_node_.Request<gz::msgs::Pose_V, gz::msgs::Boolean>(
          "/world/" + world_ + "/set_pose_vector", poses,
          [](const gz::msgs::Boolean &, const bool) {});
    }
  }

  std::string model_name(size_t i) const { return prefix_ + std::to_string(i); }

  void spawn(size_t i, double x, double y) {
    const std::string name = model_name(i);
    const std::string s = std::to_string(mesh_scale_);
    gz::msgs::EntityFactory req;
    req.set_allow_renaming(false);
    req.set_sdf("<?xml version='1.0'?><sdf version='1.6'>"
                "<model name='" +
                name +
                "'>"
                "<static>true</static>"
                "<pose>" +
                std::to_string(x) + " " + std::to_string(y) + " " +
                std::to_string(z_) + " " + std::to_string(yaw_[i]) + " 0 0" +
                "</pose>"
                "<link name='base_link'><visual name='visual'>"
                "<pose>" +
                std::to_string(mesh_offset_[0]) + " " +
                std::to_string(mesh_offset_[1]) + " " +
                std::to_string(mesh_offset_[2]) + " " +
                std::to_string(mesh_offset_[3]) + " 0 0" +

                "</pose>"
                "<geometry><mesh><uri>" +
                mesh_ +
                "</uri>"
                "<scale>" +
                s + " " + s + " " + s +
                "</scale></mesh></geometry>"
                "</visual></link></model></sdf>");

    auto logger = this->get_logger();
    std::function<void(const gz::msgs::Boolean &, const bool)> cb =
        [logger, name](const gz::msgs::Boolean &rep, const bool result) {
          if (result && rep.data()) {
            RCLCPP_INFO(logger, "Spawned %s in gz sim", name.c_str());
          } else {
            RCLCPP_ERROR(logger, "Could not spawn %s in gz sim", name.c_str());
          }
        };
    gz_node_.Request("/world/" + world_ + "/create", req, cb);
  }
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GzObstacleNode>());
  rclcpp::shutdown();
  return 0;
}
