#include <Eigen/Dense>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// ACADOS includes
#include "acados_c/ocp_nlp_interface.h"
#include "acados_solver_asv_dynamics.h"

// ROS deps
#include "rclcpp/rclcpp.hpp"

#include "asv_interfaces/msg/obstacle_list.hpp"
#include "asv_interfaces/msg/ref.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "std_msgs/msg/color_rgba.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_srvs/srv/empty.hpp"
#include "visualization_msgs/msg/marker.hpp"

#define NX ASV_DYNAMICS_NX
#define NU ASV_DYNAMICS_NU
#define NP ASV_DYNAMICS_NP

struct WeightParams {
  double min_t, max_t;
};

using namespace std::chrono_literals;

class MPCNode : public rclcpp::Node {
public:
  MPCNode() : Node("mpc_node") {
    using namespace std::placeholders;

    init_parameters();
    init_subscribers();
    init_publishers();

    timer_ = this->create_wall_timer(50ms, std::bind(&MPCNode::update, this));

    sol_path_msg.header.frame_id = frame_id;
    min_avo_path_msg.header.frame_id = frame_id;
    max_avo_path_msg.header.frame_id = frame_id;
    sol_array_msg.header.frame_id = frame_id;

    sol_path_msg.poses.resize(N_HORIZON + 1);
    min_avo_path_msg.poses.resize(n_points);
    max_avo_path_msg.poses.resize(n_points);
    sol_array_msg.poses.resize(sol_array_length);

    init_acados_solver();

    debug_weights_msg.data.resize(N_WP);
  }

  ~MPCNode() {
    // === CLEANUP ===
    asv_dynamics_acados_free(ocp_capsule);
    asv_dynamics_acados_free_capsule(ocp_capsule);
  }

private:
  static constexpr double TF = 100.0; // seconds
  static constexpr int N_HORIZON =
      ASV_DYNAMICS_N; // Assuming this macro comes from ACADOS
  static constexpr double DT = TF / N_HORIZON;
  static constexpr int N_OBS = 3;
  static constexpr int N_SP = 16; // Spline params (4 x 2 NDIMS x 2 splines)
  static constexpr int N_WP = 9;  // Weight params
  static constexpr int N_AP = 3;  // Additional params
  static constexpr int N_OP = N_OBS * 2; // Obstacle params (velocities)
  static constexpr int n_points = 20;
  static constexpr const char *frame_id = "world";
  static constexpr int sol_array_length = 10;
  static constexpr int ellipse_points = 50;

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr sol_path_pub_,
      obs_path_pub_, max_avo_path_pub_, min_avo_path_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr sol_array_pub_;
  rclcpp::Publisher<asv_interfaces::msg::Ref>::SharedPtr ref_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr sol_time_pub_,
      debug_ae_pub_, debug_ce_pub_, debug_he_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr
      debug_weights_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr ellipse_pub_,
      obs_prediction_pub_;

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr
      spline_params_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr spline_t_sub_,
      spline_t_la_sub_;
  rclcpp::Subscription<visualization_msgs::msg::Marker>::SharedPtr
      la_marker_sub_;

  rclcpp::Subscription<asv_interfaces::msg::ObstacleList>::SharedPtr
      obstacle_list_sub_;

  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr unblock_mpc_srv_;

  std::shared_ptr<rclcpp::ParameterEventHandler> weights_param_sub_,
      enabled_param_sub_, tf_param_sub_;
  std::shared_ptr<rclcpp::ParameterCallbackHandle> weights_param_handle_,
      enabled_param_handle_, tf_param_handle_;

  asv_interfaces::msg::Ref ref_msg;
  std_msgs::msg::Float64 sol_time_msg, debug_ae_msg, debug_ce_msg, debug_he_msg;
  nav_msgs::msg::Path sol_path_msg, min_avo_path_msg, max_avo_path_msg;
  geometry_msgs::msg::PoseArray sol_array_msg;
  std_msgs::msg::Float64MultiArray debug_weights_msg;

  Eigen::Vector3d nu_ref;
  Eigen::Vector3d nu_alpha{0.9, 0.9, 0.95};

  double along_e{0.0}, cross_e{0.0}, obs_d{std::numeric_limits<double>::max()};

  // w_along, w_cross, w_heading, w_input, w_surge, w_sway, w_yaw, w_terminal,
  // w_avoidance
  std::vector<double> mpc_weights{0.01,  10.0,  100.0, 0.01, 0.1,
                                  100.0, 0.001, 10.0,  0.0};
  std::vector<double> tracking_to_avoid{10.0, 0.1, 10.0, 1.0, 1.0,
                                        1.0,  1.0, 1.0,  1.0};
  std::vector<double> avoidance_weights{0.1,   1.0,   1000.0, 0.01,   0.1,
                                        100.0, 0.001, 10.0,   50000.0};

  // map input [min,max] to output [min,max]
  static constexpr double ae_start = 200.0, ae_end = 150.0;
  static constexpr double min_ce = 10.0, max_ce = 120.0;
  static constexpr double avoidance_start = 250.0, avoidance_end = 100.0;

  double tracking_weights_dynamics[N_WP]{10.0, 10.0, 5.0,  1.0, 10.0,
                                         1.0,  1.0,  10.0, 1.0};
  int warmup_count{0};
  static constexpr int WARMUP_ITERS = 5;

  WeightParams tracking_weights_inputs[N_WP]{
      // These first weights depend on separation (cross_err)
      {min_ce, max_ce}, // along
      {min_ce, max_ce}, // cross
      {min_ce, max_ce}, // heading

      // These last weights depend on remaining dist. (along_err)
      {ae_start, ae_end}, // input
      {ae_start, ae_end}, // slack
      {ae_start, ae_end}, // surge
      {ae_start, ae_end}, // yaw
      {ae_start, ae_end}, // terminal

      {ae_start, ae_end}, // avoidance
  };

  double mpc_tf{TF}, s_t{0.0};
  bool mpc_enabled{false};
  bool mpc_broken{false};
  Eigen::Vector3d asv_breakdown;
  Eigen::Vector2d asv, nearest_obs;
  double obs_predicted_d{std::numeric_limits<double>::max()};

  rclcpp::TimerBase::SharedPtr timer_;

  double ocp_params[NP];
  double pt_weights[N_WP];
  double x0[NX];

  asv_dynamics_solver_capsule *ocp_capsule;
  ocp_nlp_config *nlp_config;
  ocp_nlp_dims *nlp_dims;
  ocp_nlp_in *nlp_in;
  ocp_nlp_out *nlp_out;
  ocp_nlp_solver *nlp_solver;

  double simX[NX];
  double simU[NU];

  double xtraj[NX * (N_HORIZON + 1)];

  double last_theta{0.0};

  void update_all_params() {
    for (int i = 0; i <= N_HORIZON; i++) {
      asv_dynamics_acados_update_params(ocp_capsule, i, ocp_params, NP);
    }
  }

  void update() {
    // Params may always be changing
    recompute_weights(obs_predicted_d);
    update_all_params();

    // Update initial state
    memcpy(simX, x0, NX * sizeof(double));

    auto start_t = std::chrono::high_resolution_clock::now();

    // Set initial state constraint
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0,
                                  "lbx", simX);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0,
                                  "ubx", simX);
    int status;
    if (warmup_count < WARMUP_ITERS) {
      // Full SQP solve to build a good initial trajectory
      // (matches Python's rti_phase=0 for first 5 steps)
      int rti_phase = 0;
      ocp_nlp_solver_opts_set(nlp_config, ocp_capsule->nlp_opts, "rti_phase",
                              &rti_phase);
      status = asv_dynamics_acados_solve(ocp_capsule);
      warmup_count++;
    } else {
      // Normal RTI: preparation then feedback
      int rti_phase = 1;
      ocp_nlp_solver_opts_set(nlp_config, ocp_capsule->nlp_opts, "rti_phase",
                              &rti_phase);
      status = asv_dynamics_acados_solve(ocp_capsule);

      if (status != 0 && status != 2 && status != 5) {
        RCLCPP_WARN(this->get_logger(), "Preparation phase returned status %d",
                    status);
      }

      rti_phase = 2;
      ocp_nlp_solver_opts_set(nlp_config, ocp_capsule->nlp_opts, "rti_phase",
                              &rti_phase);
      status = asv_dynamics_acados_solve(ocp_capsule);
    }

    // Get optimal control
    ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, 0, "u", simU);

    auto end_t = std::chrono::high_resolution_clock::now();
    sol_time_msg.data = std::chrono::duration<double>(end_t - start_t).count();

    auto stamp = this->get_clock()->now();
    sol_path_msg.header.stamp = stamp;
    min_avo_path_msg.header.stamp = stamp;
    max_avo_path_msg.header.stamp = stamp;
    sol_array_msg.header.stamp = stamp;

    geometry_msgs::msg::PoseStamped tmp_pose, obs_pose;
    double min_obs_predicted_d = std::numeric_limits<double>::max();

    int stride = N_HORIZON / sol_array_length;
    for (int i = 0; i <= N_HORIZON; i++) {
      ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, i, "x", &xtraj[i * NX]);
      tmp_pose.pose.position.x = xtraj[i * NX];
      tmp_pose.pose.position.y = xtraj[i * NX + 1];
      tf2::Quaternion q;
      q.setRPY(0, 0, xtraj[i * NX + 2]);
      tmp_pose.pose.orientation = tf2::toMsg(q);

      sol_path_msg.poses[i] = tmp_pose;

      for (int j = 0; j < 3; j++) {
        obs_pose.pose.position.x = xtraj[i * NX + 7 + j * 2];
        obs_pose.pose.position.y = xtraj[i * NX + 8 + j * 2];
        double dist = distance(tmp_pose, obs_pose);
        if (dist < min_obs_predicted_d)
          min_obs_predicted_d = dist;
      }
      if (i % stride == 0 && i / stride < sol_array_length) {
        sol_array_msg.poses[i / stride] = tmp_pose.pose;
      }
    }
    publish_obs_marker(xtraj);
    obs_predicted_d = min_obs_predicted_d;

    int sol_idx = 1;
    filter_sol(Eigen::Vector3d{xtraj[sol_idx * NX + 3], xtraj[sol_idx * NX + 4],
                               xtraj[sol_idx * NX + 5]});
    ref_msg.x = xtraj[sol_idx * NX + 0];
    ref_msg.y = xtraj[sol_idx * NX + 1];
    ref_msg.psi = xtraj[sol_idx * NX + 2];
    ref_msg.u = nu_ref(0);
    ref_msg.v = nu_ref(1);
    ref_msg.r = nu_ref(2);

    debug_ce_msg.data = cross_e;
    debug_ae_msg.data = along_e;
    debug_he_msg.data = get_heading_e();
    for (int i = 0; i < N_WP; i++) {
      double w = ocp_params[N_SP + i];
      // Transform weights to log-scale for better visualization.
      debug_weights_msg.data[i] = (w > 0.0) ? std::log10(w) : -6.0;
    }

    sol_time_pub_->publish(sol_time_msg);
    sol_path_pub_->publish(sol_path_msg);
    sol_array_pub_->publish(sol_array_msg);

    if (!mpc_enabled || status == 4) {
      RCLCPP_WARN(this->get_logger(), "MPC IS DISABLED");
      if (!mpc_broken) {
        mpc_broken = true;
        asv_breakdown << x0[0], x0[1], x0[2];
      }
      ref_msg.x = asv_breakdown(0);
      ref_msg.y = asv_breakdown(1);
      ref_msg.psi = asv_breakdown(2);
      ref_msg.u = 0.0;
      ref_msg.v = 0.0;
      ref_msg.r = 0.0;

      // Re-initialize all stages to current state
      for (int i = 0; i <= N_HORIZON; i++) {
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "x", x0);
        double u_zero[NU] = {0.0, 0.0, 0.0, 0.0};
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "u", u_zero);
      }
      warmup_count = 0; // Force re-warmup

    } else {
      mpc_broken = false;
    }

    // Draw min/max avoidance radius circles around ASV
    geometry_msgs::msg::PoseStamped p;
    p.header.frame_id = frame_id;
    for (int i = 0; i < n_points; i++) {
      double angle = 2.0 * M_PI * i / (n_points - 1);

      p.pose.position.x = x0[0] + avoidance_start * std::cos(angle);
      p.pose.position.y = x0[1] + avoidance_start * std::sin(angle);
      min_avo_path_msg.poses[i] = p; // least avoidance behavior

      p.pose.position.x = x0[0] + avoidance_end * std::cos(angle);
      p.pose.position.y = x0[1] + avoidance_end * std::sin(angle);
      max_avo_path_msg.poses[i] = p; // most avoidance behavior
    }

    min_avo_path_pub_->publish(min_avo_path_msg);
    max_avo_path_pub_->publish(max_avo_path_msg);
    ref_pub_->publish(ref_msg);
    debug_ae_pub_->publish(debug_ae_msg);
    debug_ce_pub_->publish(debug_ce_msg);
    debug_he_pub_->publish(debug_he_msg);
    debug_weights_pub_->publish(debug_weights_msg);
    publish_ellipse_marker(x0[0], x0[1], x0[2]);
  }

  // Get a linear variable weight depending on t and its restrictions
  // y(t0), {t0, t1}, y(t0)*k, t[t0->t1]
  double var_w_at(double weight, WeightParams p, double dynamics, double t) {
    double w_m = (dynamics * weight - weight) / (p.max_t - p.min_t);
    double w_b = weight - w_m * p.min_t;
    if (weight < dynamics * weight)
      return std::clamp(w_m * t + w_b, weight, dynamics * weight);
    // In some cases, slope is negative, and sol. shouldn't depend on argument
    // order...
    return std::clamp(w_m * t + w_b, dynamics * weight, weight);
  }

  double interpol_at(double min_t, double max_t, double min_y, double max_y,
                     double t) {
    double w_m = (max_y - min_y) / (max_t - min_t);
    double w_b = min_y - w_m * min_t;
    if (min_y < max_y)
      return std::clamp(w_m * t + w_b, min_y, max_y);
    // In some cases, slope is negative, and sol. shouldn't depend on argument
    // order...
    return std::clamp(w_m * t + w_b, max_y, min_y);
  }

  double get_crosstrack_e() {
    Eigen::Vector2d spline_pos;
    spline_pos = get_spline(s_t);
    return distance(asv, spline_pos);
  }

  double get_heading_e() {
    Eigen::Vector2d s_dot;
    s_dot = get_spline_dot(s_t);
    double psi_ref = std::atan2(s_dot.y(), s_dot.x());
    double he_sqrt = std::sin((x0[2] - psi_ref) / 2.0);
    return he_sqrt * he_sqrt;
  }

  Eigen::Vector2d get_spline(double t) {
    return Eigen::Vector2d{ocp_params[0] * t * t * t + ocp_params[1] * t * t +
                               ocp_params[2] * t + ocp_params[3],
                           ocp_params[4] * t * t * t + ocp_params[5] * t * t +
                               ocp_params[6] * t + ocp_params[7]};
  }

  Eigen::Vector2d get_spline_dot(double t) {
    return Eigen::Vector2d{
        3 * ocp_params[0] * t * t + 2 * ocp_params[1] * t + ocp_params[2],
        3 * ocp_params[4] * t * t + 2 * ocp_params[5] * t + ocp_params[6]};
  }

  Eigen::Vector2d get_nearest_obs() {
    double min_dist = std::numeric_limits<double>::max();
    Eigen::Vector2d out, tmp;
    for (int i = 0; i < N_OBS; i++) {
      tmp << x0[7 + i * 2], x0[8 + i * 2];
      if (distance(asv, tmp) < min_dist) {
        out = tmp;
        min_dist = distance(asv, tmp);
      }
    }
    return out;
  }

  double distance(const Eigen::Vector2d &a, const Eigen::Vector2d &b) {
    return (a - b).norm();
  }

  double distance(const geometry_msgs::msg::PoseStamped &a,
                  const geometry_msgs::msg::PoseStamped &b) {
    double dx = a.pose.position.x - b.pose.position.x;
    double dy = a.pose.position.y - b.pose.position.y;
    return std::hypot(dx, dy);
  }

  void filter_sol(const Eigen::Vector3d &new_sol) {
    nu_ref = nu_alpha.cwiseProduct(nu_ref) +
             (Eigen::Vector3d::Ones() - nu_alpha).cwiseProduct(new_sol);
  }

  void publish_ellipse_marker(double asv_x, double asv_y, double asv_psi) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = this->now();
    marker.ns = "safety_ellipse";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    marker.action = visualization_msgs::msg::Marker::ADD;

    // Line thickness and color (Bright Red)
    marker.scale.x = 1.0;
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;
    marker.color.a = 0.5;

    // Effective dimensions from Python script
    double A_ELL_EFF = 95.0; // 65.0 length + safety radius
    double B_ELL_EFF = 50.0; // 20.0 width + safety radius

    for (int i = 0; i <= ellipse_points; i++) {
      // Calculate the angle for this point
      double theta = static_cast<double>(i) / ellipse_points * 2.0 * M_PI;

      // 1. Point in local ASV frame
      double lx = A_ELL_EFF * std::cos(theta);
      double ly = B_ELL_EFF * std::sin(theta);

      // 2. Rotate to global heading and translate to global position
      geometry_msgs::msg::Point p;
      p.x = asv_x + (lx * std::cos(asv_psi) - ly * std::sin(asv_psi));
      p.y = asv_y + (lx * std::sin(asv_psi) + ly * std::cos(asv_psi));
      p.z = 0.0; // Keep it flat on the water

      marker.points.push_back(p);
    }

    ellipse_pub_->publish(marker);
  }

  void publish_obs_marker(double *mpc_sol) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = this->now();
    marker.ns = "obstacle_predictions";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::LINE_LIST;
    marker.action = visualization_msgs::msg::Marker::ADD;

    // Line thickness
    marker.scale.x = 1.0;
    double r{1}, g{0.33}, b{0}, a{0.65}; // Orange
    marker.color =
        std_msgs::build<std_msgs::msg::ColorRGBA>().r(r).g(g).b(b).a(a);

    int last_sol = N_HORIZON * NX;
    geometry_msgs::msg::Point p;
    for (int i = 0; i < N_OBS; i++) {
      p.x = mpc_sol[7 + 2 * i];
      p.y = mpc_sol[8 + 2 * i];
      marker.points.push_back(p);
      p.x = mpc_sol[last_sol + 7 + 2 * i];
      p.y = mpc_sol[last_sol + 8 + 2 * i];
      marker.points.push_back(p);
    }

    obs_prediction_pub_->publish(marker);
  }

  void init_parameters() {
    this->declare_parameter("mpc_tf", mpc_tf);
    mpc_tf = this->get_parameter("mpc_tf").as_double();
    this->declare_parameter("mpc_weights", mpc_weights);
    mpc_weights = this->get_parameter("mpc_weights").as_double_array();
    this->declare_parameter("mpc_enabled", mpc_enabled);
    mpc_enabled = this->get_parameter("mpc_enabled").as_bool();
  }

  void init_subscribers() {
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/asv/state/odom", 1, [this](const nav_msgs::msg::Odometry &msg) {
          auto &q = msg.pose.pose.orientation;

          x0[0] = msg.pose.pose.position.x;
          x0[1] = msg.pose.pose.position.y;
          asv << x0[0], x0[1];

          // Feed continuous heading to MPC (remove angle wrapping)
          double new_psi = std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                                      1.0 - 2.0 * (q.y * q.y + q.z * q.z));
          double delta = new_psi - last_theta;
          if (delta > M_PI)
            x0[2] = x0[2] + (new_psi - 2.0 * M_PI) - last_theta;
          else if (delta < -M_PI)
            x0[2] = x0[2] + (new_psi + 2.0 * M_PI) - last_theta;
          else
            x0[2] = x0[2] + delta;
          last_theta = new_psi;

          x0[3] = msg.twist.twist.linear.x;
          x0[4] = msg.twist.twist.linear.y;
          x0[5] = msg.twist.twist.angular.z;
        });

    spline_t_la_sub_ = this->create_subscription<std_msgs::msg::Float64>(
        "/mpc/spline_t_la", 10, [this](const std_msgs::msg::Float64 &msg) {
          ocp_params[N_SP + N_WP] = msg.data;
        });

    la_marker_sub_ = this->create_subscription<visualization_msgs::msg::Marker>(
        "/lookahead_marker", 10,
        [this](const visualization_msgs::msg::Marker &msg) {
          Eigen::Vector2d la_pos;
          la_pos << msg.pose.position.x, msg.pose.position.y;
          along_e = distance(la_pos, asv);
        });

    spline_t_sub_ = this->create_subscription<std_msgs::msg::Float64>(
        "/mpc/spline_t", 10, [this](const std_msgs::msg::Float64 &msg) {
          s_t = fmod(msg.data, 1.0);
          x0[6] = msg.data;
          ocp_params[N_SP + N_WP + 2] = ceil(msg.data);
          cross_e = get_crosstrack_e();
        });

    spline_params_sub_ =
        this->create_subscription<std_msgs::msg::Float64MultiArray>(
            "/mpc/spline_params", 10,
            [this](const std_msgs::msg::Float64MultiArray &msg) {
              bool in_last_spline = true;
              for (int i = 0; i < 8; i++) {
                if (std::fabs(msg.data[i] - msg.data[8 + i]) > 1e-4)
                  in_last_spline = false;
              }
              ocp_params[N_SP + N_WP + 1] = static_cast<int>(in_last_spline);

              for (int i = 0; i < N_SP; i++) {
                if (std::fabs(ocp_params[i] - msg.data[i]) > 1e-6) {
                  ocp_params[i] = msg.data[i];
                }
              }
            });

    obstacle_list_sub_ =
        this->create_subscription<asv_interfaces::msg::ObstacleList>(
            "/mpc/near_obs", 10,
            [this](const asv_interfaces::msg::ObstacleList &msg) {
              if (static_cast<int>(msg.obs_list.size()) < N_OBS)
                return;
              for (int i = 0; i < N_OBS; i++) {
                x0[7 + i * 2] = msg.obs_list[i].x;
                x0[7 + 1 + i * 2] = msg.obs_list[i].y;

                int param_idx = N_SP + N_WP + N_AP;
                ocp_params[param_idx + i * 2] = msg.obs_list[i].v_x;
                ocp_params[param_idx + 1 + i * 2] = msg.obs_list[i].v_y;
              }
            });

    unblock_mpc_srv_ = this->create_service<std_srvs::srv::Empty>(
        "/mpc/unblock",
        [this](const std::shared_ptr<std_srvs::srv::Empty::Request>,
               std::shared_ptr<std_srvs::srv::Empty::Response>) {
          RCLCPP_WARN(this->get_logger(), "UNBLOCKING MPC - Resetting solver");

          // Reset solver state
          asv_dynamics_acados_reset(ocp_capsule, 1);

          // Set feasible initial trajectory (hover in place)
          for (int i = 0; i <= N_HORIZON; i++) {
            // Set all stages to current state
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "x", x0);

            // Set zero controls
            double u_zero[NU] = {0.0, 0.0, 0.0, 0.0};
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "u",
                            u_zero);
          }
          warmup_count = 0;
          RCLCPP_INFO(this->get_logger(), "Solver reset complete");
        });

    // === PARAMETER EVENT HANDLERS ===
    // For weight values
    weights_param_sub_ = std::make_shared<rclcpp::ParameterEventHandler>(this);
    auto weights_param_cb = [this](const rclcpp::Parameter &p) {
      mpc_weights = p.as_double_array();
      for (size_t i = 0; i < mpc_weights.size(); i++) {
        avoidance_weights[i] = mpc_weights[i] * tracking_to_avoid[i];
      }
      mpc_weights[8] = 0.0; // avoidance weight is zero for tracking behavior
    };
    weights_param_handle_ = weights_param_sub_->add_parameter_callback(
        "mpc_weights", weights_param_cb);

    // For MPC toggle
    enabled_param_sub_ = std::make_shared<rclcpp::ParameterEventHandler>(this);
    auto enabled_param_cb = [this](const rclcpp::Parameter &p) {
      mpc_enabled = p.as_bool();
    };
    enabled_param_handle_ = enabled_param_sub_->add_parameter_callback(
        "mpc_enabled", enabled_param_cb);

    // For TF update
    tf_param_sub_ = std::make_shared<rclcpp::ParameterEventHandler>(this);
    auto tf_param_cb = [this](const rclcpp::Parameter &p) {
      mpc_tf = p.as_double();
      double mpc_dt = mpc_tf / N_HORIZON;

      double new_time_steps[N_HORIZON];
      for (int i = 0; i < N_HORIZON; i++) {
        new_time_steps[i] = mpc_dt;
      }

      // Tear down the existing solver
      asv_dynamics_acados_free(ocp_capsule);
      asv_dynamics_acados_free_capsule(ocp_capsule);

      // Recreate with new discretization
      ocp_capsule = asv_dynamics_acados_create_capsule();
      int status = asv_dynamics_acados_create_with_discretization(
          ocp_capsule, N_HORIZON, new_time_steps);

      // Re-fetch all the handles since they point into the new capsule
      nlp_config = asv_dynamics_acados_get_nlp_config(ocp_capsule);
      nlp_dims = asv_dynamics_acados_get_nlp_dims(ocp_capsule);
      nlp_in = asv_dynamics_acados_get_nlp_in(ocp_capsule);
      nlp_out = asv_dynamics_acados_get_nlp_out(ocp_capsule);
      nlp_solver = asv_dynamics_acados_get_nlp_solver(ocp_capsule);

      // Push current params to every stage (fresh solver has defaults)
      for (int i = 0; i <= N_HORIZON; i++) {
        asv_dynamics_acados_update_params(ocp_capsule, i, ocp_params, NP);
      }

      // Force warmup so the next solve rebuilds a good trajectory
      warmup_count = 0;

      if (status != 0)
        RCLCPP_WARN(this->get_logger(),
                    "Failed to recreate solver with new Tf!");
      else
        RCLCPP_INFO(this->get_logger(), "Recreated solver: Tf=%.2fs, dt=%.4fs",
                    mpc_tf, mpc_dt);
    };
    tf_param_handle_ =
        tf_param_sub_->add_parameter_callback("mpc_tf", tf_param_cb);
  }

  void init_publishers() {
    sol_time_pub_ =
        this->create_publisher<std_msgs::msg::Float64>("/mpc/sol_time", 10);
    sol_path_pub_ =
        this->create_publisher<nav_msgs::msg::Path>("/mpc/sol_path", 10);
    obs_path_pub_ =
        this->create_publisher<nav_msgs::msg::Path>("/mpc/obs_path", 10);
    min_avo_path_pub_ =
        this->create_publisher<nav_msgs::msg::Path>("/mpc/min_avo_path", 10);
    max_avo_path_pub_ =
        this->create_publisher<nav_msgs::msg::Path>("/mpc/max_avo_path", 10);
    sol_array_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>(
        "/mpc/sol_array", 10);
    ref_pub_ =
        this->create_publisher<asv_interfaces::msg::Ref>("/asv/state/ref", 10);
    debug_ae_pub_ =
        this->create_publisher<std_msgs::msg::Float64>("/mpc/debug/a_e", 10);
    debug_ce_pub_ =
        this->create_publisher<std_msgs::msg::Float64>("/mpc/debug/c_e", 10);
    debug_he_pub_ =
        this->create_publisher<std_msgs::msg::Float64>("/mpc/debug/h_e", 10);
    debug_weights_pub_ =
        this->create_publisher<std_msgs::msg::Float64MultiArray> //
        ("/mpc/debug/w_log", 10);
    ellipse_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
        "asv_safety_ellipse", 10);
    obs_prediction_pub_ =
        this->create_publisher<visualization_msgs::msg::Marker>(
            "/mpc/obs_prediction", 10);
  }

  void init_acados_solver() {
    RCLCPP_INFO(this->get_logger(), "Creating OCP solver...");
    ocp_capsule = asv_dynamics_acados_create_capsule();
    int status = asv_dynamics_acados_create_with_discretization(
        ocp_capsule, N_HORIZON, NULL);
    if (status) {
      RCLCPP_INFO(this->get_logger(),
                  "OCP solver creation failed with status %d", status);
    } else {
      RCLCPP_INFO(this->get_logger(), "OCP solver created successfully");
    }

    nlp_config = asv_dynamics_acados_get_nlp_config(ocp_capsule);
    nlp_dims = asv_dynamics_acados_get_nlp_dims(ocp_capsule);
    nlp_in = asv_dynamics_acados_get_nlp_in(ocp_capsule);
    nlp_out = asv_dynamics_acados_get_nlp_out(ocp_capsule);
    nlp_solver = asv_dynamics_acados_get_nlp_solver(ocp_capsule);

    // Set initial state
    for (int i = 0; i < N_OBS * 2; i++) {
      x0[7 + i] = -1000.0;
    }
    memcpy(simX, x0, NX * sizeof(double));

    for (int i = 0; i < N_WP; i++) {
      ocp_params[N_SP + i] = mpc_weights[i];
    }
  }

  // Variable weights dependant on crosstrack or alongtrack errors.
  void recompute_weights(double d) {
    double alpha = interpol_at(avoidance_start, avoidance_end, 1.0, 0.0, d);
    // Path-tracking weights
    for (int i = 0; i < N_WP; i++) {
      // Get path-tracking weight
      if (i < 3)
        pt_weights[i] = var_w_at(mpc_weights[i], tracking_weights_inputs[i],
                                 tracking_weights_dynamics[i], cross_e);
      else
        pt_weights[i] = var_w_at(mpc_weights[i], tracking_weights_inputs[i],
                                 tracking_weights_dynamics[i], along_e);

      // Interpolate weights
      ocp_params[N_SP + i] =
          pt_weights[i] * alpha + avoidance_weights[i] * (1 - alpha);
    }
  }
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MPCNode>());
  rclcpp::shutdown();
  return 0;
}
