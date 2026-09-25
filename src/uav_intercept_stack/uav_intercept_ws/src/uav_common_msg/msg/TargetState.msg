# Relative state of the target with respect to the interceptor, expressed in
# the local NED frame. Populated either from ground-truth simulation state
# (target_sim) or, later, from a vision + tracking pipeline.

std_msgs/Header header

geometry_msgs/Vector3 relative_position   # target_pos - own_pos [m], NED
geometry_msgs/Vector3 relative_velocity   # target_vel - own_vel [m/s], NED

float32 range               # norm(relative_position) [m]
float32 closing_velocity    # negative range rate; positive = closing [m/s]

bool valid                  # false if target is not currently tracked
