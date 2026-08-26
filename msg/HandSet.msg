# Every hand one node saw in a single inference, and who saw them.
#
# Published by each node onto a topic they all share, so any node can draw any
# other node's hands. An empty hands[] is a positive "I currently see no hands",
# which is different from a node going silent -- receivers drop a source that
# stops publishing rather than leaving its last hand frozen on screen.
std_msgs/Header header

int32 source_id # Unique generated number at launch.
string label  # friendly name drawn on the overlay

# How this sender would like its hands drawn. a == 0 means "no preference, pick
# one for me", which is what receivers fall back to their own palette for.
std_msgs/ColorRGBA colour

Hand[] hands
