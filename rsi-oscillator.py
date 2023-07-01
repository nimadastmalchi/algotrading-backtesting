from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd

from backtesting.test import SMA, GOOG

def RSI(array, n):
    """Relative strength index"""
    # Approximate; good enough
    gain = pd.Series(array).diff()
    loss = gain.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    rs = gain.ewm(n).mean() / loss.abs().ewm(n).mean()
    return 100 - 100 / (1 + rs)

class RSIOscillator(Strategy):
    rsi_window = 10
    rsi_lower_bound = 30
    rsi_upper_bound = 85

    def init(self):
        price = self.data.Close
        self.rsi = self.I(RSI, price, self.rsi_window)

    def next(self):
        if crossover(self.rsi, self.rsi_upper_bound):
            self.sell()
        elif crossover(self.rsi_lower_bound, self.rsi):
            self.buy()

bt = Backtest(GOOG, RSIOscillator, commission=.002,
              exclusive_orders=True)

# run without optimization:
#stats = bt.run()

# run with optimization:
stats = bt.optimize(
    rsi_upper_bound = range(10, 85, 5),
    rsi_lower_bound = range(10, 85, 5),
    maximize = 'Sharpe Ratio',
    constraint = lambda param : param.rsi_upper_bound > param.rsi_lower_bound
)

bt.plot()
