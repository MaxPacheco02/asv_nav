#ifndef ASV_INTERFACES__RVIZ_TOOLS__CENTER_ON_ROBOT_TOOL_HPP_
#define ASV_INTERFACES__RVIZ_TOOLS__CENTER_ON_ROBOT_TOOL_HPP_

#include <rviz_common/tool.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

namespace asv_interfaces
{
namespace rviz_tools
{

class CenterOnRobotTool : public rviz_common::Tool
{
  Q_OBJECT

public:
  CenterOnRobotTool();
  ~CenterOnRobotTool() override = default;

  void onInitialize() override;
  void activate() override;
  void deactivate() override;
  int processKeyEvent(QKeyEvent *event, rviz_common::RenderPanel *panel) override;

private:
  void centerOnRobot();
  
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::string robot_frame_;
};

}  // namespace rviz_tools
}  // namespace asv_interfaces

#endif  // ASV_INTERFACES__RVIZ_TOOLS__CENTER_ON_ROBOT_TOOL_HPP_
