# One detected hand.
#
# Landmarks are normalized to the sender's frame (0..1 across width and height)
# rather than pixels, so a receiver with a different camera resolution can scale
# them to its own window without needing to know the sender's frame size.
# The pipeline is 2D, so z is unused.
geometry_msgs/Point[21] landmarks

# Axis-aligned box around those landmarks, normalized the same way: x, y of the
# top-left corner then width, height. Derived from the landmark extremes with a
# margin, and sent explicitly so a consumer that only wants a box does not have
# to reimplement the padding to get the same one the sender drew.
float32[4] bbox

float32 score        # presence confidence, 0..1
float32 handedness   # P(right hand); < 0.5 is a left hand
