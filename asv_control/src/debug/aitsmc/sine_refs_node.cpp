#include <asv_interfaces/msg/detail/aitsmc_debug__struct.hpp>
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
#include "asv_interfaces/msg/state.hpp"
#include "asv_interfaces/msg/thrust.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

using namespace std::chrono_literals;

class RefsPublisherNode : public rclcpp::Node {
public:
  RefsPublisherNode() : Node("refs_publisher_node") {
    this->declare_parameter("amp_u", 1.5);
    this->declare_parameter("amp_v", 0.3);
    this->declare_parameter("amp_r", 0.005);

    this->declare_parameter("freq_u", 0.05);
    this->declare_parameter("freq_v", 0.005);
    this->declare_parameter("freq_r", 0.005);

    this->declare_parameter("off_u", 3.0);
    this->declare_parameter("off_v", 0.0);
    this->declare_parameter("off_r", 0.0);

    this->declare_parameter("noise_u", 0.0);
    this->declare_parameter("noise_v", 0.0);
    this->declare_parameter("noise_r", 0.0);

    ref_pub_ =
        this->create_publisher<asv_interfaces::msg::State>("asv/state/ref", 10);

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

    asv_interfaces::msg::State ref_msg;
    ref_msg.u = sig(0) + noise_sigma(0) * gauss_(rng_);
    ref_msg.v = sig(1) + noise_sigma(1) * gauss_(rng_);
    ref_msg.r = sig(2) + noise_sigma(2) * gauss_(rng_);
    ref_msg.u_dot = sig_d(0);
    ref_msg.v_dot = sig_d(1);
    ref_msg.r_dot = sig_d(2);
    ref_pub_->publish(ref_msg);

    counter++;
  }

private:
  rclcpp::Publisher<asv_interfaces::msg::State>::SharedPtr ref_pub_;

  rclcpp::TimerBase::SharedPtr update_timer_;

  int counter{300};
  double t{0.001}; // 1 KHz
  Eigen::Vector3d amp, freq, off, noise_sigma;

  std::mt19937 rng_{std::random_device{}()};
  std::normal_distribution<double> gauss_{0.0, 1.0};

  void update_params() {
    amp(0) = this->get_parameter("amp_u").as_double();
    amp(1) = this->get_parameter("amp_v").as_double();
    amp(2) = this->get_parameter("amp_r").as_double();

    freq(0) = this->get_parameter("freq_u").as_double();
    freq(1) = this->get_parameter("freq_v").as_double();
    freq(2) = this->get_parameter("freq_r").as_double();

    off(0) = this->get_parameter("off_u").as_double();
    off(1) = this->get_parameter("off_v").as_double();
    off(2) = this->get_parameter("off_r").as_double();

    noise_sigma(0) = this->get_parameter("noise_u").as_double();
    noise_sigma(1) = this->get_parameter("noise_v").as_double();
    noise_sigma(2) = this->get_parameter("noise_r").as_double();
  }
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RefsPublisherNode>());
  rclcpp::shutdown();
  return 0;
}
