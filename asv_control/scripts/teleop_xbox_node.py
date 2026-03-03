#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from asv_interfaces.msg import Thrust
import math

def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

class TeleopControl(Node):

    def __init__(self):
        super().__init__('teleop_xbox_node')

        # Subscriptions
        self.joy_sub = self.create_subscription(
            Joy, '/joy', self.convert_joy, 10
        )

        # Publishers for left and right thruster PWM
        self.thrust_pub = self.create_publisher(Thrust, '/asv/thrust', 10)
        self.thrust_msg = Thrust()
        
        # PWM output limits (thruster pair signals)
        self.max_pwm = 2222224.5896 # maximum PWM

        self.timer = self.create_timer(0.01,
                                       self.timer_callback)
    
    def timer_callback(self):
        self.thrust_pub.publish(self.thrust_msg)
        
    def convert_joy(self, msg: Joy):
        # THRUST ANGLE
        # axes[0] and [3] go from -1 to 1, which should be inverted.
        # However, gz sim and rviz both use ENU instead of NED, so it's left as is.
        left_x = msg.axes[0]
        left_y = msg.axes[1]
        right_x = msg.axes[3]
        right_y = msg.axes[4]

        axes = [left_x, left_y, right_x, right_y]
        for i in range(len(axes)):
            if math.fabs(axes[i]) < 0.15:
                axes[i] = 0.0

        left_ang = math.atan2(axes[0], axes[1])
        right_ang = math.atan2(axes[2], axes[3])

        # THRUST FORCE
        # y = -0.5x + 0.5 = -0.5(x-1)
        left_f = -(msg.axes[2]-1.0)/2.0
        right_f = -(msg.axes[5]-1.0)/2.0

        self.thrust_msg.force0 = left_f * self.max_pwm
        self.thrust_msg.force1 = right_f * self.max_pwm
        self.thrust_msg.ang0 = left_ang
        self.thrust_msg.ang1 = right_ang

def main(args=None):
    rclpy.init(args=args)
    node = TeleopControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
