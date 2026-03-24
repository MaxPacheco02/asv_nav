#include <asv_interfaces/msg/detail/aitsmc_debug__struct.hpp>
#include <geometry_msgs/msg/detail/vector3__struct.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "asv_interfaces/msg/aitsmc_debug.hpp"
#include "asv_interfaces/msg/ref.hpp"
#include "asv_interfaces/msg/state.hpp"
#include "asv_interfaces/msg/thrust.hpp"
#include "geometry_msgs/msg/vector3.hpp"
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

    state_sub_ = this->create_subscription<asv_interfaces::msg::State>(
        "asv/state", 10, [this](const asv_interfaces::msg::State &msg) {
          asv.x = msg.x;
          asv.y = msg.y;
          asv.psi = msg.psi;
          asv.u = msg.u;
          asv.v = msg.v;
          asv.r = msg.r;
          asv.u_dot = msg.u_dot;
          asv.v_dot = msg.v_dot;
          asv.r_dot = msg.r_dot;

          odom_received = true;
        });

    reference_sub_ = this->create_subscription<asv_interfaces::msg::Ref>(
        "asv/state/ref", 10, [this](const asv_interfaces::msg::Ref &msg) {
          if (abs(pow(msg.x - asv_d.x, 2) + pow(msg.y - asv_d.y, 2)) > 30.0)
            control.reset_integral();
          // if (std::abs(msg.u - asv_d.u) > surge_threshold)
          //   control.reset_integral(0);
          // if (std::abs(msg.psi - asv_d.psi) > head_threshold)
          //   control.reset_integral(2);

          asv_d.x = msg.x;
          asv_d.y = msg.y;
          asv_d.psi = msg.psi;

          asv_d.u = 0;
          asv_d.v = 0;
          asv_d.r = 0;

          // TODO: Consider computing feedforward (from mpc sol. or spline)
          asv_d.u_dot = 0;
          asv_d.v_dot = 0;
          asv_d.r_dot = 0;

          ref_received = true;

          tf2::Quaternion q;
          q.setRPY(0, 0, msg.psi);
          ref_pose_msg.pose.position.x = msg.x;
          ref_pose_msg.pose.position.y = msg.y;
          ref_pose_msg.pose.orientation = tf2::toMsg(q);
        });

    tmp_thrust_pub_ =
        this->create_publisher<geometry_msgs::msg::Vector3>("asv/forces", 10);
    thrust_pub_ =
        this->create_publisher<asv_interfaces::msg::Thrust>("asv/thrust", 10);

    surge_debug_pub_ = this->create_publisher<asv_interfaces::msg::AitsmcDebug>(
        "aitsmc/debug/x", 10);
    sway_debug_pub_ = this->create_publisher<asv_interfaces::msg::AitsmcDebug>(
        "aitsmc/debug/y", 10);
    heading_debug_pub_ =
        this->create_publisher<asv_interfaces::msg::AitsmcDebug>(
            "aitsmc/debug/psi", 10);

    ref_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
        "/aitsmc/ref", 10);

    update_timer_ =
        this->create_wall_timer(10ms, std::bind(&AitsmcNode::update, this));

    ref_pose_msg.header.frame_id = "world";
  }

protected:
  void update() {
    if (!odom_received || !ref_received)
      return;

    Azimuth thrust = control.update(asv, asv_d);

    if (std::isnan(thrust.force0) || std::isnan(thrust.force1)) {
      RCLCPP_ERROR(this->get_logger(), "NaN thrust! Shutting down.");
      rclcpp::shutdown();
      return;
    }

    asv_interfaces::msg::Thrust thrust_msg;
    thrust_msg.force0 = thrust.force0;
    thrust_msg.force1 = thrust.force1;
    thrust_msg.ang0 = thrust.ang0;
    thrust_msg.ang1 = thrust.ang1;
    thrust_pub_->publish(thrust_msg);

    asv_interfaces::msg::AitsmcDebug surge_debug_msg;
    surge_debug_msg = debug_to_ros(control.getDebugData(0));
    surge_debug_pub_->publish(surge_debug_msg);

    asv_interfaces::msg::AitsmcDebug sway_debug_msg;
    sway_debug_msg = debug_to_ros(control.getDebugData(1));
    sway_debug_pub_->publish(sway_debug_msg);

    asv_interfaces::msg::AitsmcDebug heading_debug_msg;
    heading_debug_msg = debug_to_ros(control.getDebugData(2));
    heading_debug_pub_->publish(heading_debug_msg);

    ref_pose_msg.header.stamp = this->get_clock()->now();
    ref_pose_pub_->publish(ref_pose_msg);
  }

private:
  rclcpp::Publisher<asv_interfaces::msg::Thrust>::SharedPtr thrust_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr tmp_thrust_pub_;
  rclcpp::Publisher<asv_interfaces::msg::AitsmcDebug>::SharedPtr
      surge_debug_pub_,
      sway_debug_pub_, heading_debug_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr ref_pose_pub_;

  rclcpp::Subscription<asv_interfaces::msg::State>::SharedPtr state_sub_;
  rclcpp::Subscription<asv_interfaces::msg::Ref>::SharedPtr reference_sub_;

  geometry_msgs::msg::PoseStamped ref_pose_msg;

  rclcpp::TimerBase::SharedPtr update_timer_;

  AITSMC control;
  AITSMCParams p;
  State asv{0, 0, 0, 0, 0, 0, 0, 0, 0};   // ASV state
  State asv_d{0, 0, 0, 0, 0, 0, 0, 0, 0}; // ASV's desired state

  double surge_threshold{0.5};
  double head_threshold{0.1};

  bool odom_received{false}, ref_received{false};

  AITSMCParams initialize_params() {
    // Declare and get in one shot
    auto get_param = [&](const std::string &name) {
      this->declare_parameter(name, 0.0);
      return this->get_parameter(name).as_double();
    };

    AITSMCParams p;
    p.x = AITSMCStateParams{get_param("beta_x"),    get_param("epsilon_x"),
                            get_param("k_alpha_x"), get_param("k_beta_x"),
                            get_param("tc_x"),      get_param("q_x"),
                            get_param("p_x")};
    p.y = AITSMCStateParams{get_param("beta_y"),    get_param("epsilon_y"),
                            get_param("k_alpha_y"), get_param("k_beta_y"),
                            get_param("tc_y"),      get_param("q_y"),
                            get_param("p_y")};
    p.psi =
        AITSMCStateParams{get_param("beta_psi"),    get_param("epsilon_psi"),
                          get_param("k_alpha_psi"), get_param("k_beta_psi"),
                          get_param("tc_psi"),      get_param("q_psi"),
                          get_param("p_psi")};
    return p;
  }

  asv_interfaces::msg::AitsmcDebug debug_to_ros(const AITSMCDebugData &data) {
    asv_interfaces::msg::AitsmcDebug out;
    out.e = data.e;
    out.e_i = data.e_i;
    out.e_i_dot = data.e_i_dot;
    out.s = data.s;
    out.k = data.K;
    out.u = data.U;
    return out;
  }
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<AitsmcNode>());
  rclcpp::shutdown();
  return 0;
}
