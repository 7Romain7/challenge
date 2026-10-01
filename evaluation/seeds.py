"""Fixed seed splits (see PROTOCOL_challenge2.md section 4). TEST is run once, at the end."""

DEV = range(0, 1000)  # design, pilots, training of learned methods
VAL = range(1000, 1200)  # hyper-parameter / model selection
TEST = range(10000, 10200)  # frozen; never used for any decision

SPLITS = {"dev": DEV, "val": VAL, "test": TEST}
