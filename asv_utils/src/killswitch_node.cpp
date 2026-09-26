#include "asv_interfaces/msg/thrust.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_srvs/srv/empty.hpp"
#include <functional>

using namespace std::chrono_literals;

// Forwards the USV thrust (asv_interfaces/Thrust, force0/force1 [N]) to the
// gz sim thrusters. The USV model follows Fossen's NED convention, where
// N = 0.5 * B * (force0 - force1) and "port" thrust gives r > 0. Gazebo/ROS
// are ENU (y left, yaw CCW), where r > 0 comes from the RIGHT thruster, so
// force0 goes to the right joint and force1 to the left one.
class KillSwitchSimNode : public rclcpp::Node {
public:
  KillSwitchSimNode() : Node("killswitch_sim_node") {
    using namespace std::placeholders;

    thrust_sub_ = this->create_subscription<asv_interfaces::msg::Thrust>(
        "/usv/thrust", 10, [this](const asv_interfaces::msg::Thrust &msg) {
          if (std::isnan(msg.force0) || std::isnan(msg.force1))
            return;
          left.data = msg.force0;
          right.data = msg.force1;
          last_thrust_msg = this->get_clock()->now();
        });

    obs_twist_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel", 10, [this](const geometry_msgs::msg::Twist &msg) {
          left_obs.data = 200 * (msg.linear.x + msg.angular.z / 2);
          right_obs.data = 200 * (msg.linear.x - msg.angular.z / 2);
        });

    left_pub_ = this->create_publisher<std_msgs::msg::Float64>(
        "/model/usv/joint/left_engine_propeller_joint/cmd_thrust", 10);
    right_pub_ = this->create_publisher<std_msgs::msg::Float64>(
        "/model/usv/joint/right_engine_propeller_joint/cmd_thrust", 10);
    left_obs_pub_ = this->create_publisher<std_msgs::msg::Float64>(
        "/model/vtec_s3/joint/left_engine_propeller_joint/cmd_thrust", 10);
    right_obs_pub_ = this->create_publisher<std_msgs::msg::Float64>(
        "/model/vtec_s3/joint/right_engine_propeller_joint/cmd_thrust", 10);

    service = this->create_service<std_srvs::srv::Empty>(
        "auto", std::bind(&KillSwitchSimNode::autonomous, this, _1, _2));

    last_thrust_msg = this->get_clock()->now();
    updateTimer = this->create_wall_timer(
        10ms, std::bind(&KillSwitchSimNode::update, this));
  }

protected:
  void autonomous(const std::shared_ptr<std_srvs::srv::Empty::Request>,
                  std::shared_ptr<std_srvs::srv::Empty::Response>) {
    autonomous_on = !autonomous_on;
    RCLCPP_INFO(get_logger(), "Setting autonomous as: %s",
                autonomous_on ? "true" : "false");
  }

private:
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr service;

  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr left_pub_, right_pub_,
      left_obs_pub_, right_obs_pub_;
  rclcpp::Subscription<asv_interfaces::msg::Thrust>::SharedPtr thrust_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr obs_twist_sub_;

  std_msgs::msg::Float64 left, right, left_obs, right_obs, zero;
  rclcpp::Time last_thrust_msg;

  bool autonomous_on{false};

  rclcpp::TimerBase::SharedPtr updateTimer;

  void update() {
    left_obs_pub_->publish(right_obs);
    right_obs_pub_->publish(left_obs);

    // 200 ms of no reception: stop the USV thrusters
    if (this->get_clock()->now() - last_thrust_msg >
        rclcpp::Duration(0, 200 * 1e6)) {
      left_pub_->publish(zero);
      right_pub_->publish(zero);
      return;
    }
    left_pub_->publish(right);
    right_pub_->publish(left);
  }
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<KillSwitchSimNode>());
  rclcpp::shutdown();
  return 0;
}
