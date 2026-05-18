#include "asv_interfaces/rviz_tools/dyn_obs_tool.hpp"
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rviz_common/display_context.hpp>
#include <rviz_rendering/objects/arrow.hpp>

namespace asv_interfaces
{
namespace rviz_tools
{

DynObsTool::DynObsTool()
{
  shortcut_key_ = 'a';
  setName("Dynamic Obstacle");
}

void DynObsTool::onPoseSet(double x, double y, double theta)
{
  arrow_->setColor(1.0, 0.4, 0.0, 1.0); // RGBA orange
  std::string fixed_frame = context_->getFixedFrame().toStdString();
  geometry_msgs::msg::PoseStamped pose;
  pose.header.frame_id = fixed_frame;
  pose.header.stamp = context_->getClock()->now();
  pose.pose.position.x = x;
  pose.pose.position.y = y;
  pose.pose.position.z = 0.0;

  tf2::Quaternion quat;
  quat.setRPY(0.0, 0.0, theta);
  pose.pose.orientation.x = quat.x();
  pose.pose.orientation.y = quat.y();
  pose.pose.orientation.z = quat.z();
  pose.pose.orientation.w = quat.w();

  auto pub = context_->getRosNodeAbstraction().lock()->get_raw_node()->
    template create_publisher<geometry_msgs::msg::PoseStamped>("/rviz/dyn_obs", 1);
  pub->publish(pose);
}

}  // namespace rviz_tools
}  // namespace asv_interfaces

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(asv_interfaces::rviz_tools::DynObsTool, rviz_common::Tool)
