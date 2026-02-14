cmake_minimum_required(VERSION 3.5)
project(ardupilot_gz_bringup)

# --------------------------------------------------------------------------- #
# Find dependencies.
find_package(ament_cmake REQUIRED)

# --------------------------------------------------------------------------- #
#  Install.

# Install project launch files.
install(
  DIRECTORY
    launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install project configuration files.
install(
  DIRECTORY
    config/
  DESTINATION share/${PROJECT_NAME}/config
)

# Install project rviz files.
install(
  DIRECTORY
    rviz/
  DESTINATION share/${PROJECT_NAME}/rviz
)

# install(PROGRAMS
#   scripts/iris_mission.py
#   scripts/twist_stamped_republisher.py
#   scripts/jackal_velocity.py
#   DESTINATION lib/${PROJECT_NAME}
# )

install(PROGRAMS
  
  scripts/twist_stamped_republisher.py
  scripts/Traj_Pred_Using_Sensor_Data.py
  scripts/NMPC_Controller_v5.py
  scripts/Aruco_Detection_Node_v6.py
  scripts/Moving_the_UGV.py
  scripts/plotting_for_JP_MBK.py
  DESTINATION lib/${PROJECT_NAME}
)


# --------------------------------------------------------------------------- #
#  Build tests.

if(BUILD_TESTING)
  # Override default flake8 configuration.
  set(ament_cmake_flake8_CONFIG_FILE ${CMAKE_SOURCE_DIR}/.flake8)

  # Add linters.
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# --------------------------------------------------------------------------- #
#  Call last.



# install(DIRECTORY scripts/
#   DESTINATION lib/${PROJECT_NAME}
#   USE_SOURCE_PERMISSIONS
# )

ament_package()
