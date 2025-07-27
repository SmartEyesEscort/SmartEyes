#include <iostream>
#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;


bool bDebug = true;
/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {

  // initially set to false, set to true in first call of ProcessMeasurement
  is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;


  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // Sigma point spreading param
  lambda_ = 3 - n_aug_;

  // Initial state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units & rads
  x_ = VectorXd(n_x_);
  x_.fill(0.0);


  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);
  P_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 100, 0, 0,   // Moderate uncertainty about velocity
        0, 0, 0, 10, 0,    // Moderate uncertainty about yaw
        0, 0, 0, 0, 1;     // Lower uncertainty about yaw rate


  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  Xsig_pred_.fill(0.0);

  // Weights of sigma points
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1; i < 2 * n_aug_ + 1; ++i) {
    weights_(i) = 0.5 / (lambda_ + n_aug_);
  }

  // Process noise standard deviation (longitudinal acceleration) in m/s^2
  std_a_ = 0.3;

  // Process noise standard deviation (yaw acceleration) in rad/s^2
  std_yawdd_ = 0.3;

  // time when the state is true, in us
  time_us_ = 0.0;


  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */

}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  // Check if the UKF has been initialized with the first measurement
  if (!is_initialized_) {
    // Initialize the state vector based on the first measurement
    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      // LIDAR provides direct x,y position measurements
      // Set the state with the initial location and zero velocity
      // State vector: [px, py, v, yaw, yawd]
      x_ << meas_package.raw_measurements_[0],  // px - x position
            meas_package.raw_measurements_[1],  // py - y position
            0,                                  // v - velocity (unknown, set to 0)
            0,                                  // yaw - yaw angle (unknown, set to 0)
            0;                                  // yawd - yaw rate (unknown, set to 0)
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      // RADAR provides polar coordinates: range, bearing, range rate
      double rho = meas_package.raw_measurements_[0];      // range (distance)
      double phi = meas_package.raw_measurements_[1];      // bearing (angle)
      double rho_dot = meas_package.raw_measurements_[2];  // range rate (radial velocity)

      // Convert polar coordinates to cartesian coordinates
      // Set the state with the initial location and zero velocity
      if (fabs(rho) < 1e-6) {
        rho = 1e-6; // prevent zero magnitude
      }
      double px = rho * cos(phi);
      double py = rho * sin(phi);
      
      // Estimate velocity from range rate and bearing
      double v = fabs(rho_dot); // Use magnitude of range rate as velocity estimate
      if (v < 0.1) v = 0.1;     // Minimum velocity assumption
      
      x_ << px, py, v, phi, 0;  // Use bearing as initial yaw estimate
    }
    else {
      // Invalid sensor type - print error and exit
      std::cerr << "UKF::ProcessMeasurement() error: cannot initialize because of invalid measurement sensor type " << meas_package.sensor_type_<< std::endl;
      exit(EXIT_FAILURE);
    }

    // Store the timestamp and mark the filter as initialized
    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;

    // Debug output showing initialization
    if (bDebug)
      std::cout << "UKF::ProcessMeasurement() initializes for sensor type " << meas_package.sensor_type_ << " - state vector x:\n" << x_ << std::endl;

    return;  // Exit early after initialization
  }

  // Calculate time elapsed since last measurement
  // Convert from microseconds to seconds and add bounds check for time step
  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
  if (dt <= 0.0001) {  // Skip if time step too small or negative
    return;
  }
  if (dt > 0.5) {      // Cap maximum time step to prevent large jumps
    dt = 0.5;
  }
  
  time_us_ = meas_package.timestamp_;


  // PREDICTION STEP: Predict the state and covariance to the current time
  Prediction(dt);
  if (bDebug)
    std::cout << "UKF::ProcessMeasurement() predicts the state vector x:\n" << x_ << std::endl;

  // UPDATE STEP: Update the predicted state with the new measurement
  if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    // Use LIDAR-specific update function (linear measurement model)
    UpdateLidar(meas_package);
    if (bDebug)
      std::cout << "UKF::ProcessMeasurement() Lidar update - state vector x:\n" << x_ << std::endl;
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    // Use RADAR-specific update function (non-linear measurement model)
    UpdateRadar(meas_package);
    if (bDebug)
      std::cout << "UKF::ProcessMeasurement() Radar update - state vector x:\n" << x_ << std::endl;
  }
  else {
    // Invalid sensor type during update - print error and exit
    std::cerr << "UKF::ProcessMeasurement() error: cannot update measurement because of invalid measurement sensor type " << meas_package.sensor_type_ << std::endl;
    exit(EXIT_FAILURE);
  }

  return;
}

void UKF::Prediction(double delta_t) {
  /* 
  Estimate the object's location. Modify the state vector, x_. 
  Predict sigma points, the state, and the state covariance matrix.
  */

  /* 1. Generate augmented sigma points */
  // The augmented state includes process noise to capture uncertainty in the motion model
  // Augmented state: [px, py, v, yaw, yawd, nu_a, nu_yawdd]
  // where nu_a is longitudinal acceleration noise, nu_yawdd is yaw acceleration noise

  // Create augmented mean vector (7D: original 5D state + 2D process noise)
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.head(5) = x_;  // Copy current state estimate
  x_aug(5) = 0;        // Process noise mean for longitudinal acceleration (assumed zero)
  x_aug(6) = 0;        // Process noise mean for yaw acceleration (assumed zero)

  // Create augmented state covariance matrix (7x7)
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5, 5) = P_;                    // Current state covariance (5x5)
  P_aug(5, 5) = std_a_ * std_a_;                     // Process noise variance for acceleration
  P_aug(6, 6) = std_yawdd_ * std_yawdd_;            // Process noise variance for yaw acceleration

  // Create matrix to hold sigma points (7 x 15 for 7D augmented state)
  // We generate 2*n_aug + 1 = 15 sigma points
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // Calculate square root of covariance matrix using Cholesky decomposition
  // This gives us the "spread" directions for sigma points
  MatrixXd L = P_aug.llt().matrixL();

  // Generate augmented sigma points
  Xsig_aug.col(0) = x_aug;  // First sigma point is the mean itself
  for (int i = 0; i < n_aug_; ++i) {
    // Generate sigma points by adding/subtracting scaled eigenvectors
    // This creates a symmetric set of points around the mean
    Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);          // "Positive" sigma points
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i); // "Negative" sigma points
  }

  /* 2. Predict sigma points */
  // Apply the motion model to each sigma point to predict where they'll be after delta_t

  // Create matrix for predicted sigma points (5D state x 15 sigma points)
  MatrixXd Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // Process each sigma point through the motion model
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    // Extract current state values from this sigma point
    double px = Xsig_aug(0, i);        // x position
    double py = Xsig_aug(1, i);        // y position
    double v = Xsig_aug(2, i);         // velocity magnitude
    double yaw = Xsig_aug(3, i);       // yaw angle (heading)
    double yawd = Xsig_aug(4, i);      // yaw rate (angular velocity)
    double nu_a = Xsig_aug(5, i);      // longitudinal acceleration noise
    double nu_yawdd = Xsig_aug(6, i);  // yaw acceleration noise

    // Apply motion model equations
    double px_pred, py_pred;

    // Use different motion equations based on whether we're turning or going straight
    if (fabs(yawd) > 0.001) {
      px_pred = px + v/yawd * (sin(yaw + yawd*delta_t) - sin(yaw));
      py_pred = py + v/yawd * (-cos(yaw + yawd*delta_t) + cos(yaw));
    }
    else {
      px_pred = px + v*delta_t*cos(yaw);
      py_pred = py + v*delta_t*sin(yaw);
    }

    // For this motion model, velocity and yaw rate are assumed constant
    double v_pred = v;
    double yaw_pred = yaw + yawd * delta_t;  // Integrate yaw rate to get new yaw
    double yawd_pred = yawd;

    // Add process noise effects to the predicted state
    // These represent uncertainty in our motion model
    px_pred += 0.5 * delta_t * delta_t * cos(yaw) * nu_a;  // Acceleration noise affects position
    py_pred += 0.5 * delta_t * delta_t * sin(yaw) * nu_a;  // (quadratic in time)
    v_pred += delta_t * nu_a;                              // Acceleration noise affects velocity
    yaw_pred += 0.5 * delta_t * delta_t * nu_yawdd;        // Yaw acceleration noise affects yaw
    yawd_pred += delta_t * nu_yawdd;                       // Yaw acceleration noise affects yaw rate

    // Store predicted sigma point
    Xsig_pred_(0, i) = px_pred;
    Xsig_pred_(1, i) = py_pred;
    Xsig_pred_(2, i) = v_pred;
    Xsig_pred_(3, i) = yaw_pred;
    Xsig_pred_(4, i) = yawd_pred;
  }

  /* 3. Predict mean and covariance */
  // Combine all predicted sigma points to get predicted state and uncertainty

  // Calculate predicted state mean as weighted average of predicted sigma points
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);  // weights_ determined by UKF parameters
  }

  // Calculate predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    // Calculate difference between each sigma point and the predicted mean
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    
    // Normalize yaw angle to keep it within [-π, π] range
    // This prevents angle wrapping issues (e.g., 179° and -179° are actually close)
    while (x_diff(3) > M_PI) {
      x_diff(3) -= 2.0 * M_PI;
    }
    while (x_diff(3) < -M_PI) {
      x_diff(3) += 2.0 * M_PI;
    }

    // Accumulate weighted covariance contributions from each sigma point
    // This captures how much the sigma points spread around the predicted mean
    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }
}


void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * Use lidar data to update the belief about the object's position. 
   * Modify the state vector, x_, and covariance, P_.
  */

  // set measurement dimension, lidar can measure x and y
  int n_z = 2;
  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);

  // create a vector for incoming lidar measurement
  VectorXd z = meas_package.raw_measurements_; 
  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  /* 1. Transform sigma points into measurement space */

  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    // extract state vector elements
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);

    // measurement model
    Zsig(0, i) = px;
    Zsig(1, i) = py;
  }

  /* 2. Calculate mean, covariance */

  // calculate mean predicted measurement
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // calculate innovation covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // angle normalization
    while (z_diff(1) > M_PI) {
      z_diff(1) -= 2.0 * M_PI;
    }
    while (z_diff(1) < -M_PI) {
      z_diff(1) += 2.0 * M_PI;
    }

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z, n_z);
  R <<  std_laspx_ * std_laspx_, 0,
        0, std_laspy_  * std_laspy_;

  S = S + R;

  /* 3. Update state mean and covariance */

  // calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // angle normalization
    while (z_diff(1) > M_PI) {
      z_diff(1) -= 2.0 * M_PI;
    }
    while (z_diff(1) < -M_PI) {
      z_diff(1) += 2.0 * M_PI;
    }

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    while (x_diff(3) > M_PI) {
      x_diff(3) -= 2.0 * M_PI;
    }
    while (x_diff(3) < -M_PI) {
      x_diff(3) += 2.0 * M_PI;
    }

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // residual
  VectorXd z_diff = z - z_pred;
  // angle normalization
  while (z_diff(1) > M_PI) {
    z_diff(1) -= 2.0 * M_PI;
  }
  while (z_diff(1) < -M_PI) {
    z_diff(1) += 2.0 * M_PI;
  }

  // update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
}


void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   */

  // set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;
  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);

  // create a vector for incoming radar measurement
  VectorXd z = meas_package.raw_measurements_;
  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  /* 1. Transform sigma points into measurement space */
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    // extract state vector elements
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;

    // measurement model
    Zsig(0, i) = sqrt(px * px + py * py);
    Zsig(1, i) = atan2(py, px);
    Zsig(2, i) = (px * v1 + py * v2) / sqrt(px * px + py*py);
  }

  /* 2. Calculate mean, covariance */

  // calculate mean predicted measurement
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // calculate innovation covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // angle normalization
    while (z_diff(1) > M_PI) {
      z_diff(1) -= 2.0 * M_PI;
    }
    while (z_diff(1) < -M_PI) {
      z_diff(1) += 2.0 * M_PI;
    }

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z, n_z);
  R <<  std_radr_ * std_radr_, 0, 0,
        0, std_radphi_ * std_radphi_, 0,
        0, 0,std_radrd_ * std_radrd_;

  S = S + R;

  /* 3. Update state mean and covariance */

  // calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // angle normalization
    while (z_diff(1) > M_PI) {
      z_diff(1) -= 2.0 * M_PI;
    }
    while (z_diff(1) < -M_PI) {
      z_diff(1) += 2.0 * M_PI;
    }

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    while (x_diff(3) > M_PI) {
      x_diff(3) -= 2.0 * M_PI;
    }
    while (x_diff(3) < -M_PI) {
      x_diff(3) += 2.0 * M_PI;
    }

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // residual
  VectorXd z_diff = z - z_pred;
  // angle normalization
  while (z_diff(1) > M_PI) {
    z_diff(1) -= 2.0 * M_PI;
  }
  while (z_diff(1) < -M_PI) {
    z_diff(1) += 2.0 * M_PI;
  }

  // update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
}
