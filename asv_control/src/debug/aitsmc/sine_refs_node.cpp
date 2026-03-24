#include <asv_interfaces/msg/detail/aitsmc_debug__struct.hpp>
#include <asv_interfaces/msg/detail/ref__struct.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "asv_interfaces/msg/aitsmc_debug.hpp"
#include "asv_interfaces/msg/ref.hpp"
#include "asv_interfaces/msg/thrust.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

using namespace std::chrono_literals;

class RefsPublisherNode : public rclcpp::Node {
public:
  RefsPublisherNode() : Node("refs_publisher_node") {
    this->declare_parameter("amp_x", 0.6);
    this->declare_parameter("amp_y", 0.3);
    this->declare_parameter("amp_psi", 1.2);

    this->declare_parameter("freq_x", 0.1);
    this->declare_parameter("freq_y", 0.12);
    this->declare_parameter("freq_psi", 0.02);

    this->declare_parameter("off_x", 2.5);
    this->declare_parameter("off_y", 10.5);
    this->declare_parameter("off_psi", 0.0);

    ref_pub_ =
        this->create_publisher<asv_interfaces::msg::Ref>("asv/state/ref", 10);

    update_timer_ = this->create_wall_timer(
        10ms, std::bind(&RefsPublisherNode::update, this));
  }

protected:
  void update() {
    update_params();

    Eigen::Vector3d sig, sig_d;

    Eigen::Vector3d sin_ = (2 * M_PI * counter * t * freq).array().sin();
    Eigen::Vector3d cos_ = (2 * M_PI * counter * t * freq).array().cos();

    // y = a * sin(2pi * f * x)
    sig = amp.cwiseProduct(sin_) + off;
    // y_dot = a*2pi*f * cos(2pi * f * x)
    sig_d = (amp.cwiseProduct(2 * M_PI * freq)).cwiseProduct(cos_);

    asv_interfaces::msg::Ref ref_msg;
    ref_msg.x = sig(0);
    ref_msg.y = sig(1);
    ref_msg.psi = sig(2);
    ref_pub_->publish(ref_msg);

    counter++;
  }

private:
  rclcpp::Publisher<asv_interfaces::msg::Ref>::SharedPtr ref_pub_;

  rclcpp::TimerBase::SharedPtr update_timer_;

  int counter{0};
  double t{0.001}; // 1 KHz
  Eigen::Vector3d amp, freq, off;

  void update_params() {
    amp(0) = this->get_parameter("amp_x").as_double();
    amp(1) = this->get_parameter("amp_y").as_double();
    amp(2) = this->get_parameter("amp_psi").as_double();

    freq(0) = this->get_parameter("freq_x").as_double();
    freq(1) = this->get_parameter("freq_y").as_double();
    freq(2) = this->get_parameter("freq_psi").as_double();

    off(0) = this->get_parameter("off_x").as_double();
    off(1) = this->get_parameter("off_y").as_double();
    off(2) = this->get_parameter("off_psi").as_double();
  }
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RefsPublisherNode>());
  rclcpp::shutdown();
  return 0;
}
