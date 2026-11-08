Simplified stock market. All constants are fixed in the PRISM file: max_price=10,
max_capital=50, max_stocks=10, INIT_CAPITAL=floor(50/10)=5.

State variables (range, start value):
  buy_price         1..10, start 3
  sell_price        0..10, start 2
  capital           0..50, start 5
  stocks            0..10, start 0
  last_action_price 0..10, start 0

In every step the trader chooses one action, and in the same step the market moves both
prices. The trade always uses the prices BEFORE this move.

Actions:
- BUY is enabled only if capital >= buy_price (the guard is floor(capital/buy_price) > 0).
  With n = floor(capital/buy_price): capital becomes capital - n*buy_price,
  stocks becomes min(stocks + n, 10), last_action_price becomes buy_price.
  If stocks + n > 10, the stocks above 10 are paid for but lost.
- SELL is enabled only if stocks > 0. capital becomes min(capital + stocks*sell_price, 50),
  stocks becomes 0, last_action_price becomes sell_price. Capital above 50 is lost.
- HOLD is always enabled. capital, stocks and last_action_price stay the same.

Market move (both prices move together, each clamped to its range,
i.e. buy_price stays in 1..10 and sell_price in 0..10):
  after BUY : both +1 with probability 0.375, unchanged 0.375, both -1 0.25
  after SELL: both +1 with probability 0.25,  unchanged 0.375, both -1 0.375
  after HOLD: both +1 with probability 0.375, unchanged 0.25,  both -1 0.375

Labels: "success" holds exactly when capital = 50. "bankruptcy" holds exactly when
stocks = 0 and capital = 0; in such a state only HOLD is enabled. No state is terminal:
the process continues after "success" is reached.
