cmake_minimum_required(VERSION 3.10)
project(mujoco-wrapper)

# find dependencies
find_package(ament_cmake REQUIRED)

# Install targets
install(PROGRAMS
  node/position.py
  node/mocap_teleop.py
  node/collision_teleop.py
  node/joint_command_executor.py
  test/imu_test.py
  test/teleop_test.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()