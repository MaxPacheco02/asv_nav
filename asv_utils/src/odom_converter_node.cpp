#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2_ros/transform_broadcaster.h"
#include "geometry_msgs/msg/pose2_d.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_srvs/srv/empty.hpp"
#include <functional>

#include <tf2/LinearMath/Quaternion.h>

#include <cstdio>
#include <cmath>
#include <algorithm>

#include "geometry_msgs/msg/pose2_d.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64.hpp"
#include "tf2_ros/transform_broadcaster.h"

#include "asv_interfaces/msg/state.hpp"

using namespace std::chrono_literals;

class OdomConverterNode : public rclcpp::Node {
public:
  OdomConverterNode() : Node("odom_converter_node"){
    using namespace std::placeholders;

    convertedOdomPub = 
      this->create_publisher<nav_msgs::msg::Odometry>("/usv/state/odom", 10);

    odomSub = this->create_subscription<nav_msgs::msg::Odometry>("/gz_sim/odometry", 10,
      std::bind(&OdomConverterNode::odom_cb, this, _1));

    posePub = this->create_publisher<geometry_msgs::msg::Pose2D>("/usv/state/pose", 10);

    velocityPub = this->create_publisher<geometry_msgs::msg::Vector3>("/usv/state/velocity", 10);

    statePub = this->create_publisher<asv_interfaces::msg::State>("/usv/state", 10);

    // vtec_s3 (used as an obstacle), brought into the same zeroed frame
    obsOdomSub = this->create_subscription<nav_msgs::msg::Odometry>("/vtec_s3/odometry", 10,
      std::bind(&OdomConverterNode::obs_odom_cb, this, _1));

    obsOdomPub = this->create_publisher<nav_msgs::msg::Odometry>("/vtec_s3/state/odom", 10);

    tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);


    service = this->create_service<std_srvs::srv::Empty>("/zero_imu", std::bind(&OdomConverterNode::zero_imu, this, _1, _2));
  }

protected:
  // Shifts the vtec_s3 pose by the USV zero and rotates its body-frame twist
  // into the world frame (child_frame_id = "world"), as obstacle velocities
  // are world-frame.
  void obs_odom_cb(const nav_msgs::msg::Odometry::SharedPtr msg){
    // The zero comes from the USV odometry; skip until it is set
    if(!hasZeroInit)
      return;

    auto &q_msg = msg->pose.pose.orientation;
    double psi = std::atan2(2.0 * (q_msg.w * q_msg.z + q_msg.x * q_msg.y),
                            1.0 - 2.0 * (q_msg.y * q_msg.y + q_msg.z * q_msg.z));
    double u = msg->twist.twist.linear.x;
    double v = msg->twist.twist.linear.y;

    nav_msgs::msg::Odometry obsOdom = *msg;
    obsOdom.header.frame_id = "world";
    obsOdom.child_frame_id = "world";
    obsOdom.pose.pose.position.x = msg->pose.pose.position.x - zero_x;
    obsOdom.pose.pose.position.y = msg->pose.pose.position.y - zero_y;
    obsOdom.twist.twist.linear.x = u * std::cos(psi) - v * std::sin(psi);
    obsOdom.twist.twist.linear.y = u * std::sin(psi) + v * std::cos(psi);
    obsOdomPub->publish(obsOdom);
  }

  void odom_cb(const nav_msgs::msg::Odometry::SharedPtr msg){
    if(!hasZeroInit){
      zero_x = msg->pose.pose.position.x;
      zero_y = msg->pose.pose.position.y;
      hasZeroInit = true;
      RCLCPP_INFO(get_logger(), "Setting x: %f y: %f as new zero", zero_x, zero_y);
    }

    nav_msgs::msg::Odometry newOdom = *msg;
    tf2::Quaternion q(newOdom.pose.pose.orientation.x, newOdom.pose.pose.orientation.y,
                      newOdom.pose.pose.orientation.z, newOdom.pose.pose.orientation.w);
    tf2::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
    double psi = yaw;

    newOdom.header.stamp = msg->header.stamp;
    newOdom.header.frame_id = "world";
    newOdom.child_frame_id = "usv";

    newOdom.pose.pose.position.x = msg->pose.pose.position.x - zero_x;
    newOdom.pose.pose.position.y = msg->pose.pose.position.y - zero_y;

    double u_orig = msg->twist.twist.linear.x;
    double v_orig = msg->twist.twist.linear.y;

    newOdom.twist.twist.linear.x = u_orig;
    newOdom.twist.twist.linear.y = v_orig;

    newOdom.twist.twist.angular.z = msg->twist.twist.angular.z;

    convertedOdomPub->publish(newOdom);
    last_odom = newOdom;
    {
      geometry_msgs::msg::Pose2D pose;
      pose.x = newOdom.pose.pose.position.x;
      pose.y = newOdom.pose.pose.position.y;
      pose.theta = psi;
      posePub->publish(pose);
      tf_broadcast(pose);

      geometry_msgs::msg::Vector3 velocity;
      velocity.x = newOdom.twist.twist.linear.x;
      velocity.y = newOdom.twist.twist.linear.y;
      velocity.z = newOdom.twist.twist.angular.z;
      velocityPub->publish(velocity);

      // Accelerations are not measured (AITSMC_SSY does not use them)
      statePub->publish(asv_interfaces::build<asv_interfaces::msg::State>()
                            .x(pose.x)
                            .y(pose.y)
                            .psi(psi)
                            .u(velocity.x)
                            .v(velocity.y)
                            .r(velocity.z)
                            .u_dot(0.0)
                            .v_dot(0.0)
                            .r_dot(0.0));
    }
  }

  void tf_broadcast(const geometry_msgs::msg::Pose2D &msg) {
    geometry_msgs::msg::TransformStamped t;

    // Read message content and assign it to
    // corresponding tf variables
    t.header.stamp = this->get_clock()->now();
    t.header.frame_id = "world";
    t.child_frame_id = "usv";

    // Turtle only exists in 2D, thus we get x and y translation
    // coordinates from the message and set the z coordinate to 0
    t.transform.translation.x = msg.x;
    t.transform.translation.y = msg.y;
    t.transform.translation.z = 0.0;

    // For the same reason, turtle can only rotate around one axis
    // and this why we set rotation in x and y to 0 and obtain
    // rotation in z axis from the message
    tf2::Quaternion q;
    q.setRPY(0, 0, msg.theta);
    t.transform.rotation.x = q.x();
    t.transform.rotation.y = q.y();
    t.transform.rotation.z = q.z();
    t.transform.rotation.w = q.w();

    // Send the transformation
    tf_broadcaster->sendTransform(t);
  }


  void zero_imu(const std::shared_ptr<std_srvs::srv::Empty::Request>,
              std::shared_ptr<std_srvs::srv::Empty::Response>) {
    zero_x = last_odom.pose.pose.position.x + zero_x;
    zero_y = last_odom.pose.pose.position.y + zero_y;
    RCLCPP_INFO(get_logger(), "Setting x: %f y: %f as new zero", zero_x, zero_y);
  }

private:
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr service;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr convertedOdomPub;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odomSub, obsOdomSub;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr obsOdomPub;
  rclcpp::Publisher<geometry_msgs::msg::Pose2D>::SharedPtr posePub;
  rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr velocityPub;
  rclcpp::Publisher<asv_interfaces::msg::State>::SharedPtr statePub;

  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

  nav_msgs::msg::Odometry last_odom;

  bool hasZeroInit{false};
  double zero_x{0}, zero_y{0};
};

int main(int argc, char *argv[]){
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<OdomConverterNode>());
  rclcpp::shutdown();
  return 0;
}