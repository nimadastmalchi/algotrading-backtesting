from backtesting import Backtest, Strategy
from backtesting.lib import crossover

from backtesting.test import SMA, GOOG


class DualMovingAverage(Strategy):
    upper_bound = 50
    lower_bound = 10

    def init(self):
        price = self.data.Close
        self.ma1 = self.I(SMA, price, self.lower_bound)
        self.ma2 = self.I(SMA, price, self.upper_bound)

    def next(self):
        if crossover(self.ma1, self.ma2):
            self.buy()
        elif crossover(self.ma2, self.ma2):
            self.sell()

bt = Backtest(GOOG, DualMovingAverage, commission=.002,
              exclusive_orders=True)

# run without optimization:
#stats = bt.run()

# run with optimization:
stats = bt.optimize(
    upper_bound = range(10, 85, 5),
    lower_bound = range(10, 85, 5),
    maximize = 'Sharpe Ratio',
    constraint = lambda param : param.upper_bound > param.lower_bound
)

bt.plot()
