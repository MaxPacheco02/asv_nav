#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "asv_interfaces/msg/thrust.hpp"
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

#include "asv_control/model/dynamic_model.h"

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
          auto &q = msg->pose.pose.orientation;
          model = DynamicModel{
              {msg->pose.pose.position.x, msg->pose.pose.position.y,
               std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                          1.0 - 2.0 * (q.y * q.y + q.z * q.z))}};
        });

    thrust_sub_ = this->create_subscription<asv_interfaces::msg::Thrust>(
        "asv/thrust", 10, [this](const asv_interfaces::msg::Thrust &msg) {
          if (std::isnan(msg.force0) || std::isnan(msg.force1) ||
              std::isnan(msg.ang0) || std::isnan(msg.ang1)) {
            return;
          }
          thrust_ = Azimuth{msg.force0, msg.force1, msg.ang0, msg.ang1};
          last_thrust_msg = this->get_clock()->now();
        });

    pose_pub_ = this->create_publisher<geometry_msgs::msg::Pose2D>(
        "asv/state/pose", 10);
    asv_state_pub_ = this->create_publisher<asv_interfaces::msg::State>(
        "asv/state", 10);
    local_vel_pub_ = this->create_publisher<geometry_msgs::msg::Vector3>(
        "asv/state/velocity", 10);
    odom_pub_ =
        this->create_publisher<nav_msgs::msg::Odometry>("asv/state/odom", 10);
    pose_path_pub_ =
        this->create_publisher<nav_msgs::msg::Path>("asv/pose_path", 10);
    azimuth_conf_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>(
        "/azimuth/conf", 10);

    tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    odom.header.frame_id = "world";
    azimuth_conf_msg.header =      //
        pose_stamped_tmp_.header = //
        pose_path.header =         //
        odom.header;
    last_thrust_msg = this->get_clock()->now();

    update_timer_ = this->create_wall_timer(
        10ms, std::bind(&DynamicModelNode::update, this));
  }

protected:
  void update() {
    // 200 ms of no reception
    if (this->get_clock()->now() - last_thrust_msg >
        rclcpp::Duration(0, 200 * 1e6)) {
      thrust_ = Azimuth{0, 0, 0, 0};
    }

    State out = model.update(thrust_);
    
    asv_interfaces::msg::State asv_state_msg;
    asv_state_msg = asv_interfaces::build<asv_interfaces::msg::State>()
      .x(out.x).y(out.y).psi(out.psi).u(out.u).v(out.v).r(out.r)
      .u_dot(out.u_dot).v_dot(out.v_dot).r_dot(out.r_dot);

    geometry_msgs::msg::Pose2D pose;
    pose.x = out.x;
    pose.y = out.y;
    pose.theta = out.psi;
    eta << out.x, out.y, out.psi;
    odom.pose.pose = v2p(eta);

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

    pose_path.header.stamp =            //
        odom.header.stamp =             //
        azimuth_conf_msg.header.stamp = //
        this->get_clock()->now();

    azimuth_conf_msg.poses.clear();
    if (thrust_.force0 > 1e-3)
      azimuth_conf_msg.poses.push_back(
          v2p(rotate(forward(eta, model.lx0), thrust_.ang0)));
    if (thrust_.force1 > 1e-3)
      azimuth_conf_msg.poses.push_back(
          v2p(rotate(forward(eta, model.lx1), thrust_.ang1)));


    pose_pub_->publish(pose);
    odom_pub_->publish(odom);
    asv_state_pub_->publish(asv_state_msg);
    local_vel_pub_->publish(velMsg);
    pose_path_pub_->publish(pose_path);
    azimuth_conf_pub_->publish(azimuth_conf_msg);
    tf_broadcast(odom);
  }

private:
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr azimuth_conf_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Pose2D>::SharedPtr pose_pub_;
  rclcpp::Publisher<asv_interfaces::msg::State>::SharedPtr asv_state_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr local_vel_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pose_path_pub_;

  rclcpp::TimerBase::SharedPtr update_timer_;

  rclcpp::Time last_tport_msg, last_tstbd_msg, last_thrust_msg;

  geometry_msgs::msg::PoseStamped pose_stamped_tmp_;
  geometry_msgs::msg::PoseArray azimuth_conf_msg;
  nav_msgs::msg::Path pose_path;
  nav_msgs::msg::Odometry odom;

  rclcpp::Subscription<asv_interfaces::msg::Thrust>::SharedPtr thrust_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
      initial_pose_sub_;

  Azimuth thrust_;
  Eigen::Vector3d eta;
  DynamicModel model{Eigen::Vector3d{0, 0, 0}};

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

  Eigen::Vector3d forward(const Eigen::Vector3d &p, double d) {
    Eigen::Vector3d out;
    out << std::cos(p.z()), std::sin(p.z()), 0.0;
    return p + d * out;
  }

  Eigen::Vector3d rotate(const Eigen::Vector3d &p, double ang) {
    return Eigen::Vector3d{p.x(), p.y(), model.wrap_angle(p.z() + ang)};
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
  rclcpp::spin(std::make_shared<DynamicModelNode>());
  rclcpp::shutdown();
  return 0;
}
