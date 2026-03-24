#include <geometry_msgs/msg/detail/pose__struct.hpp>
#include <geometry_msgs/msg/detail/pose_stamped__struct.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "asv_interfaces/msg/ref.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "rclcpp/rclcpp.hpp"

using namespace std::chrono_literals;

class RVizRefsNode : public rclcpp::Node {
public:
  RVizRefsNode() : Node("rviz_refs_node") {
    goal_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/goal_pose", 1,
        [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
          asv_interfaces::msg::Ref ref;

          auto &q = msg->pose.orientation;
          ref.x = msg->pose.position.x;
          ref.y = msg->pose.position.y;
          ref.psi = std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                               1.0 - 2.0 * (q.y * q.y + q.z * q.z));
          ref_pub_->publish(ref);
        });

    ref_pub_ =
        this->create_publisher<asv_interfaces::msg::Ref>("asv/state/ref", 10);
  }

private:
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr
      goal_pose_sub_;
  rclcpp::Publisher<asv_interfaces::msg::Ref>::SharedPtr ref_pub_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RVizRefsNode>());
  rclcpp::shutdown();
  return 0;
}
