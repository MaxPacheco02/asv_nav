#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "asv_interfaces/msg/ref.hpp"
#include "asv_interfaces/msg/thrust.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

#include "asv_control/control/AITSMC.h"
#include "asv_control/model/dynamic_model.h"

using namespace std::chrono_literals;

class AitsmcNode : public rclcpp::Node {
public:
  AitsmcNode() : Node("aitsmc_node") {
    using namespace std::placeholders;

    p = initialize_params();
    control = AITSMC(p);

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "asv/state/odom", 10, [this](const nav_msgs::msg::Odometry &msg) {
          asv.x = msg.pose.pose.position.x;
          asv.y = msg.pose.pose.position.y;
          auto &q = msg.pose.pose.orientation;
          asv.psi = std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                               1.0 - 2.0 * (q.y * q.y + q.z * q.z));
          asv.u = msg.twist.twist.linear.x;
          asv.v = msg.twist.twist.linear.y;
          asv.r = msg.twist.twist.angular.z;
          odom_received = true;
        });

    reference_sub_ = this->create_subscription<asv_interfaces::msg::Ref>(
        "asv/state/ref", 10, [this](const asv_interfaces::msg::Ref &msg) {
          // Some references are equal to the state because they are not used.
          asv_d.x = asv.x;
          asv_d.y = asv.y;
          asv_d.psi = msg.psi;

          asv_d.u = msg.u;
          asv_d.v = 0;
          asv_d.r = 0;

          // TODO: Compute feedforward (from mpc sol. or spline)
          asv_d.u_dot = 0;
          asv_d.v_dot = 0;
          asv_d.r_dot = 0;

          ref_received = true;
        });

    thrust_pub_ =
        this->create_publisher<asv_interfaces::msg::Thrust>("asv/thrust", 10);

    update_timer_ =
        this->create_wall_timer(10ms, std::bind(&AitsmcNode::update, this));
  }

protected:
  void update() {
    if(!odom_received || !ref_received)
      return;

    Azimuth thrust = control.update(asv, asv_d);

    if (std::isnan(thrust.force0) || std::isnan(thrust.force1)) {
      RCLCPP_ERROR(this->get_logger(), "NaN thrust! Shutting down.");
      rclcpp::shutdown();
      return;
    }

    asv_interfaces::msg::Thrust msg;
    msg.force0 = thrust.force0;
    msg.force1 = thrust.force1;
    msg.ang0 = thrust.ang0;
    msg.ang1 = thrust.ang1;
    thrust_pub_->publish(msg);
  }

private:
  rclcpp::Publisher<asv_interfaces::msg::Thrust>::SharedPtr thrust_pub_;

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<asv_interfaces::msg::Ref>::SharedPtr reference_sub_;

  rclcpp::TimerBase::SharedPtr update_timer_;

  AITSMC control;
  AITSMCParams p;
  State asv{0,0,0,0,0,0,0,0,0};   // ASV state
  State asv_d{0,0,0,0,0,0,0,0,0}; // ASV's desired state

  bool odom_received{false},
       ref_received{false};

  AITSMCParams initialize_params() {
    // Declare and get in one shot
    auto get_param = [&](const std::string &name) {
      this->declare_parameter(name, 0.0);
      return this->get_parameter(name).as_double();
    };

    AITSMCParams p;
    p.epsilon_u = get_param("epsilon_u");
    p.epsilon_psi = get_param("epsilon_psi");
    p.k_alpha_u = get_param("k_alpha_u");
    p.k_alpha_psi = get_param("k_alpha_psi");
    p.k_beta_u = get_param("k_beta_u");
    p.k_beta_psi = get_param("k_beta_psi");
    p.tc_u = get_param("tc_u");
    p.tc_psi = get_param("tc_psi");
    p.q_u = get_param("q_u");
    p.q_psi = get_param("q_psi");
    p.p_u = get_param("p_u");
    p.p_psi = get_param("p_psi");
    p.beta_psi = get_param("beta_psi");
    return p;
  }
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<AitsmcNode>());
  rclcpp::shutdown();
  return 0;
}
