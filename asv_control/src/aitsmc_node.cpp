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

#include "asv_control/control/AITSMC_SSY.h"
#include "asv_control/control/AITSMC_XYH.h"
#include "asv_control/model/dynamic_model.h"

using namespace std::chrono_literals;

enum class ControlMode { XYH, SSY };

class AitsmcNode : public rclcpp::Node {
public:
  AitsmcNode() : Node("aitsmc_node") {
    using namespace std::placeholders;

    this->declare_parameter("control_mode", "ssy");
    auto mode_str = this->get_parameter("control_mode").as_string();
    if (mode_str == "xyh") {
      mode_ = ControlMode::XYH;
      control = AITSMC_XYH(initialize_xyh_params());
    } else {
      mode_ = ControlMode::SSY;
      control = AITSMC_SSY(initialize_ssy_params());
    }

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
          if (mode_ == ControlMode::XYH) {
            if (std::abs(pow(msg.x - asv_d.x, 2) + pow(msg.y - asv_d.y, 2)) >
                30.0)
              std::visit([](auto &c) { c.reset_integral(); }, control);

            asv_d.x = msg.x;
            asv_d.y = msg.y;
            asv_d.psi = msg.psi;

            asv_d.u = 0;
            asv_d.v = 0;
            asv_d.r = 0;
          } else {
            asv_d.x = 0;
            asv_d.y = 0;
            asv_d.psi = 0;

            asv_d.u = msg.u;
            asv_d.v = msg.v;
            asv_d.r = msg.r;
          }

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

    thrust_pub_ =
        this->create_publisher<asv_interfaces::msg::Thrust>("asv/thrust", 10);

    ref_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
        "/aitsmc/ref", 10);

    update_timer_ =
        this->create_wall_timer(10ms, std::bind(&AitsmcNode::update, this));

    ref_pose_msg.header.frame_id = "world";

    debug_pubs_ = {this->create_publisher<asv_interfaces::msg::AitsmcDebug>(
                       "aitsmc/debug/x", 10),
                   this->create_publisher<asv_interfaces::msg::AitsmcDebug>(
                       "aitsmc/debug/y", 10),
                   this->create_publisher<asv_interfaces::msg::AitsmcDebug>(
                       "aitsmc/debug/psi", 10)};
  }

protected:
  void update() {
    if (!odom_received || !ref_received)
      return;

    Azimuth thrust =
        std::visit([&](auto &c) { return c.update(asv, asv_d); }, control);

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

    for (int i = 0; i < 3; i++) {
      auto data =
          std::visit([i](auto &c) { return c.getDebugData(i); }, control);
      debug_pubs_[i]->publish(debug_to_ros(data));
    }

    ref_pose_msg.header.stamp = this->get_clock()->now();
    ref_pose_pub_->publish(ref_pose_msg);
  }

private:
  rclcpp::Publisher<asv_interfaces::msg::Thrust>::SharedPtr thrust_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr ref_pose_pub_;

  rclcpp::Subscription<asv_interfaces::msg::State>::SharedPtr state_sub_;
  rclcpp::Subscription<asv_interfaces::msg::Ref>::SharedPtr reference_sub_;

  geometry_msgs::msg::PoseStamped ref_pose_msg;

  rclcpp::TimerBase::SharedPtr update_timer_;

  std::array<rclcpp::Publisher<asv_interfaces::msg::AitsmcDebug>::SharedPtr, 3>
      debug_pubs_;

  ControlMode mode_{ControlMode::SSY};
  std::variant<AITSMC_XYH, AITSMC_SSY> control;
  State asv{0, 0, 0, 0, 0, 0, 0, 0, 0};   // ASV state
  State asv_d{0, 0, 0, 0, 0, 0, 0, 0, 0}; // ASV's desired state

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

  AITSMC_XYH_Params initialize_xyh_params() {
    return {read_axis_params("x"), read_axis_params("y"),
            read_axis_params("psi")};
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
