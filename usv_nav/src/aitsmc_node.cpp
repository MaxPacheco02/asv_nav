#include <tf2/LinearMath/Quaternion.h>

#include <chrono>
#include <cmath>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "asv_interfaces/msg/aitsmc_debug.hpp"
#include "asv_interfaces/msg/state.hpp"
#include "asv_interfaces/msg/thrust.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "rclcpp/rclcpp.hpp"

#include "asv_control/control/AITSMC_SSY.h"
#include "usv_nav/usv_dynamic_model.h"

using namespace std::chrono_literals;

// Velocity (surge, yaw) AITSMC for the differential-thrust USV. Sway is not
// actuated, so it is not tracked; only SSY mode is supported.
class AitsmcNode : public rclcpp::Node {
public:
  AitsmcNode() : Node("aitsmc_node") {
    control = AITSMC_SSY(initialize_ssy_params());
    control.set_model(std::make_shared<UsvDynamicModel>());

    state_sub_ = this->create_subscription<asv_interfaces::msg::State>(
        "usv/state", 10, [this](const asv_interfaces::msg::State &msg) {
          usv.x = msg.x;
          usv.y = msg.y;
          usv.psi = msg.psi;
          usv.u = msg.u;
          usv.v = msg.v;
          usv.r = msg.r;
          usv.u_dot = msg.u_dot;
          usv.v_dot = msg.v_dot;
          usv.r_dot = msg.r_dot;

          odom_received = true;
        });

    reference_sub_ = this->create_subscription<asv_interfaces::msg::State>(
        "/usv/state/ref", 10, [this](const asv_interfaces::msg::State &msg) {
          usv_d.u = msg.u;
          usv_d.r = msg.r;

          // TODO: Consider computing feedforward (from mpc sol. or spline)
          usv_d.u_dot = 0;
          usv_d.v_dot = 0;
          usv_d.r_dot = 0;

          ref_received = true;

          tf2::Quaternion q;
          q.setRPY(0, 0, msg.psi);
          ref_pose_msg.pose.position.x = msg.x;
          ref_pose_msg.pose.position.y = msg.y;
          ref_pose_msg.pose.orientation = tf2::toMsg(q);
        });

    thrust_pub_ =
        this->create_publisher<asv_interfaces::msg::Thrust>("usv/thrust", 10);

    ref_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
        "/aitsmc/ref", 10);

    update_timer_ =
        this->create_wall_timer(10ms, std::bind(&AitsmcNode::update, this));

    ref_pose_msg.header.frame_id = "world";

    debug_pubs_ = {this->create_publisher<asv_interfaces::msg::AitsmcDebug>(
                       "/aitsmc/debug/u", 10),
                   this->create_publisher<asv_interfaces::msg::AitsmcDebug>(
                       "/aitsmc/debug/v", 10),
                   this->create_publisher<asv_interfaces::msg::AitsmcDebug>(
                       "/aitsmc/debug/r", 10)};
  }

protected:
  void update() {
    if (!odom_received || !ref_received)
      return;

    // Unactuated sway: zero error so it does not drive the control law
    usv_d.v = usv.v;
    Eigen::Vector3d tau = control.compute_tau(usv, usv_d);
    auto [T_port, T_stbd] = UsvDynamicModel::allocate(tau(0), tau(2));

    // Flag axes the allocation had to scale down, for anti-windup
    Eigen::Vector3d applied{T_port + T_stbd, 0.0,
                            0.5 * UsvDynamicModel::B * (T_port - T_stbd)};
    Eigen::Vector3d sat = Eigen::Vector3d::Zero();
    for (int i : {0, 2})
      if (std::abs(applied(i)) < std::abs(tau(i)) - 1e-6)
        sat(i) = tau(i) > 0 ? 1.0 : -1.0;
    control.set_saturation(sat);

    if (std::isnan(T_port) || std::isnan(T_stbd)) {
      RCLCPP_ERROR(this->get_logger(), "NaN thrust! Shutting down.");
      rclcpp::shutdown();
      return;
    }

    asv_interfaces::msg::Thrust thrust_msg;
    thrust_msg.force0 = T_port;
    thrust_msg.force1 = T_stbd;
    thrust_pub_->publish(thrust_msg);

    for (int i = 0; i < 3; i++) {
      debug_pubs_[i]->publish(debug_to_ros(control.getDebugData(i)));
    }

    ref_pose_msg.header.stamp = this->get_clock()->now();
    ref_pose_pub_->publish(ref_pose_msg);
  }

private:
  rclcpp::Publisher<asv_interfaces::msg::Thrust>::SharedPtr thrust_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr ref_pose_pub_;

  rclcpp::Subscription<asv_interfaces::msg::State>::SharedPtr state_sub_;
  rclcpp::Subscription<asv_interfaces::msg::State>::SharedPtr reference_sub_;

  geometry_msgs::msg::PoseStamped ref_pose_msg;

  rclcpp::TimerBase::SharedPtr update_timer_;

  std::array<rclcpp::Publisher<asv_interfaces::msg::AitsmcDebug>::SharedPtr, 3>
      debug_pubs_;

  AITSMC_SSY control;
  State usv{0, 0, 0, 0, 0, 0, 0, 0, 0};   // USV state
  State usv_d{0, 0, 0, 0, 0, 0, 0, 0, 0}; // USV's desired state

  bool odom_received{false}, ref_received{false};

  AITSMCStateParams read_axis_params(const std::string &axis) {
    auto get = [&](const std::string &name) {
      this->declare_parameter(name, 0.0);
      return this->get_parameter(name).as_double();
    };
    return AITSMCStateParams{get("beta_" + axis),    get("epsilon_" + axis),
                             get("k_alpha_" + axis), get("k_beta_" + axis),
                             get("tc_" + axis),      get("q_" + axis),
                             get("p_" + axis)};
  }

  AITSMC_SSY_Params initialize_ssy_params() {
    return {read_axis_params("u"), read_axis_params("v"),
            read_axis_params("r")};
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
