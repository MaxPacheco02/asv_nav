#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "asv_control/model/dynamic_model.h"
#include "asv_interfaces/msg/thrust.hpp"
#include "geometry_msgs/msg/pose2_d.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

using namespace std::chrono_literals;

class DynamicModelNode : public rclcpp::Node {
 public:
  DynamicModelNode() : Node("dynamic_model_node") {
    using namespace std::placeholders;

    initial_pose_sub_ = this->create_subscription<
        geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/initialpose", 1,
        [this](const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr
                   msg) {
          auto& q = msg->pose.pose.orientation;
          model = DynamicModel{
              {msg->pose.pose.position.x, msg->pose.pose.position.y,
               std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                          1.0 - 2.0 * (q.y * q.y + q.z * q.z))}};
        });

    thrust_sub_ = this->create_subscription<asv_interfaces::msg::Thrust>(
        "asv/thrust", 10, [this](const asv_interfaces::msg::Thrust& msg) {
          thrust_ = Azimuth{msg.force0, msg.force1, msg.ang0,
                            msg.ang1};
          last_thrust_msg = this->get_clock()->now();
        });

    pose_pub_ = this->create_publisher<geometry_msgs::msg::Pose2D>(
        "usv/state/pose", 10);
    local_vel_pub_ = this->create_publisher<geometry_msgs::msg::Vector3>(
        "usv/state/velocity", 10);
    odom_pub_ =
        this->create_publisher<nav_msgs::msg::Odometry>("usv/state/odom", 10);
    pose_path_pub_ =
        this->create_publisher<nav_msgs::msg::Path>("usv/pose_path", 10);

    tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    odom.header.frame_id = "world";
    pose_stamped_tmp_.header = pose_path.header = odom.header;

    update_timer_ = this->create_wall_timer(
        10ms, std::bind(&DynamicModelNode::update, this));

    last_thrust_msg = this->get_clock()->now();
  }

 protected:
  void update() {
    // 200 ms of no reception
    if (this->get_clock()->now() - last_thrust_msg >
        rclcpp::Duration(0, 200 * 1e6)) {
      thrust_ = Azimuth{0, 0, 0, 0};
    }

    State out = model.update(thrust_);

    geometry_msgs::msg::Pose2D pose;
    pose.x = out.x;
    pose.y = out.y;
    pose.theta = out.psi;

    tf2::Quaternion q;
    q.setRPY(0, 0, out.psi);

    odom.pose.pose.position.x = out.x;
    odom.pose.pose.position.y = out.y;
    odom.pose.pose.orientation = tf2::toMsg(q);

    geometry_msgs::msg::Vector3 velMsg;
    odom.twist.twist.linear.x = velMsg.x = out.u;
    odom.twist.twist.linear.y = velMsg.y = out.v;
    odom.twist.twist.angular.z = velMsg.z = out.r;

    pose_stamped_tmp_.pose = odom.pose.pose;
    pose_path.poses.push_back(pose_stamped_tmp_);
    if (pose_path.poses.size() > 1000) {
      pose_path.poses.erase(pose_path.poses.begin(),
                            pose_path.poses.begin() + 1);
    }

    pose_path.header.stamp = this->get_clock()->now();
    odom.header.stamp = this->get_clock()->now();

    pose_pub_->publish(pose);
    odom_pub_->publish(odom);
    local_vel_pub_->publish(velMsg);
    pose_path_pub_->publish(pose_path);
    tf_broadcast(odom);
  }

 private:
  rclcpp::Publisher<geometry_msgs::msg::Pose2D>::SharedPtr pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr local_vel_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pose_path_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;

  rclcpp::TimerBase::SharedPtr update_timer_;

  rclcpp::Time last_tport_msg, last_tstbd_msg, last_thrust_msg;

  geometry_msgs::msg::PoseStamped pose_stamped_tmp_;
  nav_msgs::msg::Path pose_path;
  nav_msgs::msg::Odometry odom;

  rclcpp::Subscription<asv_interfaces::msg::Thrust>::SharedPtr thrust_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
      initial_pose_sub_;

  Azimuth thrust_;

  DynamicModel model{Eigen::Vector3d{0, 0, 0}};

  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

  void tf_broadcast(const nav_msgs::msg::Odometry& msg) {
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = this->get_clock()->now();
    t.header.frame_id = "world";
    t.child_frame_id = "usv";

    t.transform.translation.x = msg.pose.pose.position.x;
    t.transform.translation.y = msg.pose.pose.position.y;
    t.transform.rotation = msg.pose.pose.orientation;
    tf_broadcaster->sendTransform(t);
  }
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DynamicModelNode>());
  rclcpp::shutdown();
  return 0;
}