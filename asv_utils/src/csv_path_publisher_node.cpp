#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"

using namespace std::chrono_literals;

/*
 * Reads a k-sim log CSV with columns (at minimum):
 *     Time, Surge Speed, Sway Speed, Yaw Rate, Roll Angle, Roll Rate,
 *     Heading, x, y, Rudder Angle, Wind Direction, Wind Speed
 *
 * and publishes the (x, y, Heading) trajectory as nav_msgs/Path so it
 * can be overlayed against the dynamic model's /asv/pose_path in RViz.
 *
 * Params:
 *   csv_file    (string) absolute path to the CSV
 *   frame_id    (string) frame the path is expressed in (default "world")
 *   publish_hz  (double) re-publish rate for the (static) path
 */
class CsvPathPublisher : public rclcpp::Node {
public:
  CsvPathPublisher() : Node("csv_path_publisher") {
    const auto csv_file = this->declare_parameter<std::string>("csv_file", "");
    const auto frame_id =
        this->declare_parameter<std::string>("frame_id", "world");
    const auto publish_hz = this->declare_parameter<double>("publish_hz", 1.0);

    if (csv_file.empty()) {
      RCLCPP_FATAL(get_logger(), "Parameter 'csv_file' is required.");
      throw std::runtime_error("csv_file not set");
    }

    path_.header.frame_id = frame_id;
    if (!load_csv(csv_file, frame_id)) {
      RCLCPP_FATAL(get_logger(), "Failed to load CSV: %s", csv_file.c_str());
      throw std::runtime_error("csv load failed");
    }
    RCLCPP_INFO(get_logger(), "Loaded %zu poses from %s", path_.poses.size(),
                csv_file.c_str());

    path_pub_ = create_publisher<nav_msgs::msg::Path>("asv/csv_path", 10);

    const auto period = std::chrono::duration<double>(1.0 / publish_hz);
    timer_ = create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(period), [this]() {
          path_.header.stamp = this->get_clock()->now();
          for (auto &p : path_.poses)
            p.header.stamp = path_.header.stamp;
          path_pub_->publish(path_);
        });
  }

private:
  bool load_csv(const std::string &file, const std::string &frame_id) {
    std::ifstream in(file);
    if (!in.is_open())
      return false;

    std::string line;
    if (!std::getline(in, line))
      return false; // header
    const auto cols = split(line, ',');

    const int ix = index_of(cols, "x");
    const int iy = index_of(cols, "y");
    const int ipsi = index_of(cols, "Heading");
    if (ix < 0 || iy < 0 || ipsi < 0) {
      RCLCPP_ERROR(get_logger(),
                   "CSV missing required columns (need x, y, Heading).");
      return false;
    }

    while (std::getline(in, line)) {
      if (line.empty())
        continue;
      const auto f = split(line, ',');
      if (static_cast<int>(f.size()) <= std::max({ix, iy, ipsi}))
        continue;

      geometry_msgs::msg::PoseStamped ps;
      ps.header.frame_id = frame_id;
      ps.pose.position.x = std::stod(f[ix]);
      ps.pose.position.y = std::stod(f[iy]);

      tf2::Quaternion q;
      q.setRPY(0, 0, std::stod(f[ipsi]));
      ps.pose.orientation = tf2::toMsg(q);

      path_.poses.push_back(ps);
    }
    return !path_.poses.empty();
  }

  static std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, delim)) {
      // strip trailing \r from CRLF files
      if (!tok.empty() && tok.back() == '\r')
        tok.pop_back();
      out.push_back(tok);
    }
    return out;
  }

  static int index_of(const std::vector<std::string> &v,
                      const std::string &name) {
    for (size_t i = 0; i < v.size(); ++i)
      if (v[i] == name)
        return static_cast<int>(i);
    return -1;
  }

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  nav_msgs::msg::Path path_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CsvPathPublisher>());
  rclcpp::shutdown();
  return 0;
}
