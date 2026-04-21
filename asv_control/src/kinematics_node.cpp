#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

#include "asv_interfaces/msg/ref.hpp"
#include "asv_interfaces/msg/state.hpp"
#include "asv_interfaces/msg/thrust.hpp"
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

class KinematicsNode : public rclcpp::Node {
public:
  KinematicsNode() : Node("kinematics_node") {
    using namespace std::placeholders;

    initial_pose_sub_ = this->create_subscription<
        geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/initialpose", 1,
        [this](const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr
                   msg) {
          auto &q = msg->pose.pose.orientation;
          eta << msg->pose.pose.position.x, msg->pose.pose.position.y,
              std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z));
        });

    input_sub_ = this->create_subscription<asv_interfaces::msg::Ref>(
        "asv/state/ref", 10, [this](const asv_interfaces::msg::Ref &msg) {
          last_nu_msg = this->get_clock()->now();
          nu << msg.u, msg.v, msg.r;
        });

    pose_pub_ = this->create_publisher<geometry_msgs::msg::Pose2D>(
        "asv/state/pose", 10);
    asv_state_pub_ =
        this->create_publisher<asv_interfaces::msg::State>("asv/state", 10);
    local_vel_pub_ = this->create_publisher<geometry_msgs::msg::Vector3>(
        "asv/state/velocity", 10);
    odom_pub_ =
        this->create_publisher<nav_msgs::msg::Odometry>("asv/state/odom", 10);
    pose_path_pub_ =
        this->create_publisher<nav_msgs::msg::Path>("asv/pose_path", 10);

    tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    odom.header.frame_id = "world";
    pose_stamped_tmp_.header = //
        pose_path.header =     //
        odom.header;
    last_nu_msg = this->get_clock()->now();

    update_timer_ =
        this->create_wall_timer(10ms, std::bind(&KinematicsNode::update, this));
  }

protected:
  void update() {
    // 200 ms of no reception
    if (this->get_clock()->now() - last_nu_msg >
        rclcpp::Duration(0, 200 * 1e6)) {
      nu << 0, 0, 0;
    }

    J << std::cos(eta(2)), -std::sin(eta(2)), 0, std::sin(eta(2)),
        std::cos(eta(2)), 0, 0, 0, 1;

    eta_dot = J * nu; // transformation into local reference frame
    eta = integral_step * (eta_dot + eta_dot_last) / 2 + eta; // integral
    eta_dot_last = eta_dot;

    asv_state_msg.x = eta.x();
    asv_state_msg.y = eta.x();
    asv_state_msg.psi = eta.z();
    asv_state_msg.u = nu.x();
    asv_state_msg.v = nu.y();
    asv_state_msg.r = nu.z();

    geometry_msgs::msg::Pose2D pose;
    pose.x = eta.x();
    pose.y = eta.y();
    pose.theta = eta.z();
    odom.pose.pose = v2p(eta);

    geometry_msgs::msg::Vector3 velMsg;
    odom.twist.twist.linear.x = velMsg.x = nu.x();
    odom.twist.twist.linear.y = velMsg.y = nu.y();
    odom.twist.twist.angular.z = velMsg.z = nu.z();

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

    pose_pub_->publish(pose);
    odom_pub_->publish(odom);
    asv_state_pub_->publish(asv_state_msg);
    local_vel_pub_->publish(velMsg);
    pose_path_pub_->publish(pose_path);
    tf_broadcast(odom);
  }

private:
  rclcpp::Publisher<geometry_msgs::msg::Pose2D>::SharedPtr pose_pub_;
  rclcpp::Publisher<asv_interfaces::msg::State>::SharedPtr asv_state_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr local_vel_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pose_path_pub_;

  rclcpp::TimerBase::SharedPtr update_timer_;

  rclcpp::Time last_nu_msg;

  geometry_msgs::msg::PoseStamped pose_stamped_tmp_;
  nav_msgs::msg::Path pose_path;
  nav_msgs::msg::Odometry odom;
  asv_interfaces::msg::State asv_state_msg;

  int path_count{0};

  rclcpp::Subscription<asv_interfaces::msg::Ref>::SharedPtr input_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
      initial_pose_sub_;

  Eigen::Matrix3d J;
  Eigen::Vector3d eta{0, 0, 0}, nu{0, 0, 0};
  Eigen::Vector3d eta_dot{0, 0, 0}, eta_dot_last{0, 0, 0};
  double integral_step{0.01};

  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

  void tf_broadcast(const nav_msgs::msg::Odometry &msg) {
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = this->get_clock()->now();
    t.header.frame_id = "world";
    t.child_frame_id = "asv";

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
  rclcpp::spin(std::make_shared<KinematicsNode>());
  rclcpp::shutdown();
  return 0;
}
