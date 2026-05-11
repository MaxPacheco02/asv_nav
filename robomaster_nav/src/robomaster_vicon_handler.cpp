#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <geometry_msgs/msg/detail/pose_stamped__struct.hpp>
#include <nav_msgs/msg/detail/odometry__struct.hpp>
#include <random>

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
#include "robomaster_msgs/msg/led_effect.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/transform_broadcaster.h>

using namespace std::chrono_literals;

// Six colours to cycle through, in order.
// Each row is {r, g, b} at full brightness; the breathing envelope scales them.
static constexpr int N_COLORS = 6;
static constexpr float COLORS[N_COLORS][3] = {
    {0.4f, 0.2f, 0.8f}, // blue-purple  (user's fav, boosted b for LEDs)
    {0.0f, 0.0f, 1.0f}, // pure blue
    {1.0f, 0.0f, 0.0f}, // pure red
    {0.0f, 1.0f, 0.0f}, // pure green
    {0.0f, 1.0f, 1.0f}, // electric cyan
    {1.0f, 0.0f, 1.0f}, // magenta
};

// Ticks at 50 Hz for one full breath (0 → max → 0).
// 50 ticks = 1 s per colour.
static constexpr int PERIOD_TICKS = 75;

class RMViconHandler : public rclcpp::Node {
public:
  RMViconHandler() : Node("robomaster_vicon_handler") {
    using namespace std::placeholders;

    vicon_pose_sub_ =
        this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/vicon/RobomasterS1_2/RobomasterS1_2", 1,
            [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
              // Sometimes the vicon sends position at 0,0 when it loses the
              // odometry fix to the vehicle.
              if (msg->pose.position.x == 0 && msg->pose.position.y == 0 &&
                  eta.x() != 0 && eta.y() != 0) {
                return;
              }

              odom.header = msg->header;
              odom.pose.pose = msg->pose;
              auto &q = msg->pose.orientation;
              eta << msg->pose.position.x, msg->pose.position.y,
                  std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y * q.y + q.z * q.z));
            });

    rm_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/odom", 1, [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
          odom.twist = msg->twist;
        });

    asv_state_pub_ =
        this->create_publisher<asv_interfaces::msg::State>("asv/state", 10);
    odom_pub_ =
        this->create_publisher<nav_msgs::msg::Odometry>("asv/state/odom", 10);
    pose_path_pub_ =
        this->create_publisher<nav_msgs::msg::Path>("asv/pose_path", 10);
    area_limits_pub_ =
        this->create_publisher<nav_msgs::msg::Path>("/vicon/area_limits", 10);
    led_pub_ = this->create_publisher<robomaster_msgs::msg::LEDEffect>(
        "/leds/effect", 10);

    tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    odom.header.frame_id = "world";
    pose_stamped_tmp_.header = //
        pose_path.header =     //
        area_limits.header =   //
        odom.header;
    odom.child_frame_id = "base_link";

    for (int i = 0; i < area_lims_N; i++) {
      pose_stamped_tmp_.pose.position.x = area_lims[i * 2];
      pose_stamped_tmp_.pose.position.y = area_lims[i * 2 + 1];
      area_limits.poses.push_back(pose_stamped_tmp_);
    }

    update_timer_ =
        this->create_wall_timer(20ms, std::bind(&RMViconHandler::update, this));
  }

protected:
  void update() {
    asv_state_msg.x = eta.x();
    asv_state_msg.y = eta.y();
    asv_state_msg.psi = eta.z();
    asv_state_msg.u = odom.twist.twist.linear.x;
    asv_state_msg.v = odom.twist.twist.linear.y;
    asv_state_msg.r = odom.twist.twist.angular.z;

    pose_stamped_tmp_.pose = odom.pose.pose;
    if (path_count % 5 == 0) {
      pose_path.poses.push_back(pose_stamped_tmp_);
    }
    if (pose_path.poses.size() > 500) {
      pose_path.poses.erase(pose_path.poses.begin(),
                            pose_path.poses.begin() + 1);
    }
    path_count++;

    pose_path.header.stamp =       //
        odom.header.stamp =        //
        area_limits.header.stamp = //
        this->get_clock()->now();

    odom_pub_->publish(odom);
    asv_state_pub_->publish(asv_state_msg);
    pose_path_pub_->publish(pose_path);
    area_limits_pub_->publish(area_limits);
    tf_broadcast(odom);
    update_leds();
  }

private:
  rclcpp::Publisher<asv_interfaces::msg::State>::SharedPtr asv_state_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pose_path_pub_,
      area_limits_pub_;
  rclcpp::Publisher<robomaster_msgs::msg::LEDEffect>::SharedPtr led_pub_;

  rclcpp::TimerBase::SharedPtr update_timer_;

  geometry_msgs::msg::PoseStamped pose_stamped_tmp_;
  nav_msgs::msg::Path pose_path, area_limits;
  nav_msgs::msg::Odometry odom;
  asv_interfaces::msg::State asv_state_msg;

  int path_count{0};
  Eigen::Vector3d eta{0, 0, 0};
  int area_lims_N{8};
  double area_lims[16]{
      -4.18982437301333,   -1.5185002579609614, -4.28981351852417,
      -1.5170269012451172, -6.2754411697387695, 0.22072847187519073,
      -6.426323890686035,  2.0404179096221924,  4.514416217803955,
      2.2635483741760254,  4.651918411254883,   -1.6103342771530151,
      4.551942798658694,   -1.612542644427015,  -4.18982437301333,
      -1.5185002579609614};

  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr
      vicon_pose_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr rm_odom_sub_;

  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

  // LED state
  int led_tick_{0};
  int color_idx_{0};

  void update_leds() {
    int cycle_tick = led_tick_ % PERIOD_TICKS;

    // Advance colour at the start of every new period
    if (cycle_tick == 0 && led_tick_ > 0)
      color_idx_ = (color_idx_ + 1) % N_COLORS;

    led_tick_++;

    // Raised-cosine ease: smooth 0 → 1 interpolation from current to next
    // colour
    int next_idx = (color_idx_ + 1) % N_COLORS;
    float phase = static_cast<float>(cycle_tick) / PERIOD_TICKS;
    float alpha = (1.0f - std::cos(static_cast<float>(M_PI) * phase)) / 2.0f;

    float r =
        COLORS[color_idx_][0] * (1.0f - alpha) + COLORS[next_idx][0] * alpha;
    float g =
        COLORS[color_idx_][1] * (1.0f - alpha) + COLORS[next_idx][1] * alpha;
    float b =
        COLORS[color_idx_][2] * (1.0f - alpha) + COLORS[next_idx][2] * alpha;

    robomaster_msgs::msg::LEDEffect msg;
    msg.effect = 1;
    msg.color.r = r;
    msg.color.g = g;
    msg.color.b = b;
    msg.color.a = 1.0f;
    led_pub_->publish(msg);
  }

  void tf_broadcast(const nav_msgs::msg::Odometry &msg) {
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = this->get_clock()->now();
    t.header.frame_id = "world";
    t.child_frame_id = "base_link";

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
  rclcpp::spin(std::make_shared<RMViconHandler>());
  rclcpp::shutdown();
  return 0;
}
