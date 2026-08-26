# One detected hand.

# The point is normalized to the sender's frame (0..1 across width and height)
# rather than pixels, so a receiver with a different camera resolution can scale
# them to its own window without needing to know the sender's frame size.
# The pipeline is 2D, so z is unused.
geometry_msgs/Point point

float32 score        # presence confidence, 0..1
float32 handedness   # P(right hand); < 0.5 is a left hand
