#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;


FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  // measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  // measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

  // projection matrix - laser
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  // set the acceleration noise components
  noise_ax = 9;
  noise_ay = 9;

}

FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    cout << "EKF: " << endl;

    // initialize state convariance matrix
    Eigen::MatrixXd P_init(4,4);
    P_init << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1000, 0,
        0, 0, 0, 1000;

    // initialize state transition matrix
    Eigen::MatrixXd F_init(4,4);
    F_init << 1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1;


    // initialize process covariance matrix
    Eigen::MatrixXd Q_init(4,4);

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {

      // Convert radar from polar to cartesian coordinates and initialize state.
      float ro = measurement_pack.raw_measurements_[0];
      float phi = measurement_pack.raw_measurements_[1];
      Eigen::VectorXd x_init(4);
      x_init << ro * cos(phi), ro * sin(phi), 0, 0;

      ekf_.Init(x_init, P_init, F_init, Hj_, R_radar_, Q_init);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {

        // initialize state for laser
      Eigen::VectorXd x_init(4);
      x_init << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;

      ekf_.Init(x_init, P_init, F_init, H_laser_, R_laser_, Q_init);
    }

    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  // elapsed time
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
  previous_timestamp_ = measurement_pack.timestamp_;

  // update state transition matrix
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  // update process covariance matrix Q
  ekf_.Q_ << (pow(dt, 4) / 4) * noise_ax, 0, (pow(dt, 3) / 2) * noise_ax, 0,
          0, pow(dt, 4) / 4 * noise_ay, 0, (pow(dt, 3) / 2) * noise_ay,
          (pow(dt, 3) / 2) * noise_ax, 0, pow(dt, 2) * noise_ax, 0,
          0, (pow(dt, 3) / 2) * noise_ay, 0, pow(dt, 2) * noise_ay;

  ekf_.Predict();


  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {

    // Radar updates
    Hj_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.H_ = Hj_;
    ekf_.R_ = R_radar_;

    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {

    // Laser updates
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;

    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
