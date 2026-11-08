Frozen lake on a 4x4 grid (constants control=0.33, start_position=0).

The single input `pos` encodes the tile: pos = 4*row + col (row 0 = top, col 0 = left);
pos = 16 means "episode over".

    row 0:  0  1  2  3      S F F F
    row 1:  4  5  6  7      F H F H
    row 2:  8  9 10 11      F F F H
    row 3: 12 13 14 15      H F F G

Holes: 5, 7, 11, 12. Goal (frisbee): 15. Start: pos = start_position (0 in our runs).
a3_up moves to row-1, a1_down to row+1, a0_left to col-1, a2_right to col+1.
Slippery ice: the intended move happens with probability `control`; with probability
(1-control)/2 each, the agent instead moves in one of the two perpendicular directions.
A move into the border keeps the position. On a hole or on the goal tile every action
leads to pos=16.
