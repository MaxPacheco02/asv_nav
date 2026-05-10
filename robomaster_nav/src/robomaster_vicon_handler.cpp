#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <geometry_msgs/msg/detail/pose_stamped__struct.hpp>
#include <nav_msgs/msg/detail/odometry__struct.hpp>
#include <random>

#include "asv_interfaces/msg/state.hpp"
#include "geometry_msgs/msg/pose2_d.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/transform_broadcaster.h>

using namespace std::chrono_literals;

class RMViconHandler : public rclcpp::Node {
public:
  RMViconHandler() : Node("robomaster_vicon_handler") {
    using namespace std::placeholders;

    vicon_pose_sub_ =
        this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/vicon/RobomasterS1_2/RobomasterS1_2", 1,
            [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
              odom.header = msg->header;
              odom.pose.pose = msg->pose;
              auto &q = msg->pose.orientation;
              eta << msg->pose.position.x, msg->pose.position.y,
                  std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y * q.y + q.z * q.z));
            });

    rm_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/odom", 1, [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
          odom.twist = msg->twist;
        });

    asv_state_pub_ =
        this->create_publisher<asv_interfaces::msg::State>("asv/state", 10);
    odom_pub_ =
        this->create_publisher<nav_msgs::msg::Odometry>("asv/state/odom", 10);
    pose_path_pub_ =
        this->create_publisher<nav_msgs::msg::Path>("asv/pose_path", 10);

    tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    odom.header.frame_id = "world";
    pose_stamped_tmp_.header = //
        pose_path.header =     //
        odom.header;
    odom.child_frame_id = "base_link";

    update_timer_ =
        this->create_wall_timer(10ms, std::bind(&RMViconHandler::update, this));
  }

protected:
  void update() {
    asv_state_msg.x = eta.x();
    asv_state_msg.y = eta.y();
    asv_state_msg.psi = eta.z();
    asv_state_msg.u = odom.twist.twist.linear.x;
    asv_state_msg.v = odom.twist.twist.linear.y;
    asv_state_msg.r = odom.twist.twist.angular.z;

    pose_stamped_tmp_.pose = odom.pose.pose;
    // Record one pose per second...
    if (path_count % 100 == 0)
      pose_path.poses.push_back(pose_stamped_tmp_);
    // Record the last 1000 seconds
    if (pose_path.poses.size() > 1000) {
      pose_path.poses.erase(pose_path.poses.begin(),
                            pose_path.poses.begin() + 1);
    }
    path_count++;

    pose_path.header.stamp = //
        odom.header.stamp =  //
        this->get_clock()->now();

    odom_pub_->publish(odom);
    asv_state_pub_->publish(asv_state_msg);
    pose_path_pub_->publish(pose_path);
    tf_broadcast(odom);
  }

private:
  rclcpp::Publisher<asv_interfaces::msg::State>::SharedPtr asv_state_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pose_path_pub_;

  rclcpp::TimerBase::SharedPtr update_timer_;

  geometry_msgs::msg::PoseStamped pose_stamped_tmp_;
  nav_msgs::msg::Path pose_path;
  nav_msgs::msg::Odometry odom;
  asv_interfaces::msg::State asv_state_msg;

  int path_count{0};
  Eigen::Vector3d eta{0, 0, 0};

  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr
      vicon_pose_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr rm_odom_sub_;

  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

  void tf_broadcast(const nav_msgs::msg::Odometry &msg) {
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = this->get_clock()->now();
    t.header.frame_id = "world";
    t.child_frame_id = "base_link";

    t.transform.translation.x = msg.pose.pose.position.x;
    t.transform.translation.y = msg.pose.pose.position.y;
    t.transform.rotation = msg.pose.pose.orientation;
    tf_broadcaster->sendTransform(t);
  }

  geometry_msgs::msg::Pose v2p(const Eigen::Vector3d &vec) {
    geometry_msgs::msg::Pose out;
    tf2::Quaternion q;
    q.setRPY(0, 0, vec.z());

    out.position.x = vec.x();
    out.position.y = vec.y();
    out.orientation = tf2::toMsg(q);
    return out;
  }
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RMViconHandler>());
  rclcpp::shutdown();
  return 0;
}
