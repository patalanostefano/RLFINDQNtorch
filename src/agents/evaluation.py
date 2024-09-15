import numpy as np
from scipy import stats


class Evaluation:
    def __init__(self, data, action_label, initial_investment, trading_cost_ratio=0.001):

        self.data = data
        self.initial_investment = initial_investment
        self.action_label = action_label
        self.trading_cost_ratio = trading_cost_ratio

    def evaluate(self):

        time_weighted_return = self.time_weighted_return()
        total_return = self.total_return()
        portfolio = self.get_daily_portfolio_value()

        print('#' * 50)
        print(f'Time weighted return: {time_weighted_return}')
        print('#' * 50)
        print(f'Total Return: {total_return} %')
        print('#' * 50)
        print(f'Initial Investment: {self.initial_investment}')
        print('#' * 50)
        print(f'Final Portfolio Value: {portfolio[-1]}')
        print('#' * 50)

    def arithmetic_daily_return(self):
        # TODO: should we consider the days we have bought the share or we should act in general?
        # TODO: 1 + arithemtic_return
        self.arithmetic_return()
        return self.data[f'arithmetic_daily_return_{self.action_label}'].sum()

    def logarithmic_daily_return(self):
        self.logarithmic_return()
        return self.data[f'logarithmic_daily_return_{self.action_label}'].sum()

    def average_daily_return(self):
        # TODO 1 + arithemtic return
        self.arithmetic_return()
        return self.data[f'arithmetic_daily_return_{self.action_label}'].mean()

    def daily_return_variance(self, daily_return_type="arithmetic"):
        if daily_return_type == 'arithmetic':
            self.arithmetic_return()
            return self.data[f'arithmetic_daily_return_{self.action_label}'].var()
        elif daily_return_type == "logarithmic":
            self.logarithmic_return()
            return self.data[f'logarithmic_daily_return_{self.action_label}'].var()

    def time_weighted_return(self):
        rate_of_return = self.get_rate_of_return()
        mult = 1
        for i in rate_of_return:
            mult = mult * (i + 1)
        return np.power(mult, 1 / len(rate_of_return)) - 1

    def total_return(self):
        portfolio_value = self.get_daily_portfolio_value()
        return (portfolio_value[-1] - self.initial_investment) / self.initial_investment * 100

    def logarithmic_return(self):
        self.data[f'logarithmic_daily_return_{self.action_label}'] = 0.0

        own_share = False
        for i in range(len(self.data)):
            if self.data[self.action_label][i] == 'buy' or (own_share and self.data[self.action_label][i] == 'None'):
                own_share = True
                if i < len(self.data) - 1:
                    self.data[f'logarithmic_daily_return_{self.action_label}'][i] = np.log(
                        self.data['close'][i] / self.data['close'][i + 1])
            elif self.data[self.action_label][i] == 'sell':
                own_share = False

        self.data[f'logarithmic_daily_return_{self.action_label}'] = self.data[
            f'logarithmic_daily_return_{self.action_label}'] * 100

    def arithmetic_return(self):
        self.data[f'arithmetic_daily_return_{self.action_label}'] = 0.0

        own_share = False
        for i in range(len(self.data)):
            if (self.data[self.action_label][i] == 'buy') or (own_share and self.data[self.action_label][i] == 'None'):
                own_share = True
                if i < len(self.data) - 1:
                    self.data[f'arithmetic_daily_return_{self.action_label}'][i] = (self.data['close'][i + 1] -
                                                                                    self.data['close'][i]) / \
                        self.data['close'][i]

            elif self.data[self.action_label][i] == 'sell':
                own_share = False

        self.data[f'arithmetic_daily_return_{self.action_label}'] = self.data[
            f'arithmetic_daily_return_{self.action_label}'] * 100
        # else you have sold your share, so you would not lose or earn any more money

    def get_daily_portfolio_value(self):
        portfolio_value = [self.initial_investment]
        self.arithmetic_return()

        arithmetic_return = self.data[f'arithmetic_daily_return_{self.action_label}'] / 100
        num_shares = 0

        for i in range(len(self.data)):
            action = self.data[self.action_label][i]
            if action == 'buy' and num_shares == 0:  # then buy and pay the transaction cost
                num_shares = portfolio_value[-1] * (1 - self.trading_cost_ratio) / \
                    self.data.iloc[i]['close']
                if i + 1 < len(self.data):
                    portfolio_value.append(
                        num_shares * self.data.iloc[i + 1]['close'])

            elif action == 'sell' and num_shares > 0:  # then sell and pay the transaction cost
                portfolio_value.append(
                    num_shares * self.data.iloc[i]['close'] * (1 - self.trading_cost_ratio))
                num_shares = 0

            elif (action == 'None' or action == 'buy') and num_shares > 0:  # hold shares and get profit
                profit = arithmetic_return[i] * \
                    portfolio_value[len(portfolio_value) - 1]
                portfolio_value.append(portfolio_value[-1] + profit)

            elif (action == 'sell' or action == 'None') and num_shares == 0:
                portfolio_value.append(portfolio_value[-1])

        return portfolio_value

    def get_rate_of_return(self):
        portfolio = self.get_daily_portfolio_value()
        rate_of_return = [(portfolio[p + 1] - portfolio[p]) / portfolio[p]
                          for p in range(len(portfolio) - 1)]
        return rate_of_return

    def calculate_match_actions(self, human_actions, agent_actions):
        match = 0
        total = 0
        for i in range(len(self.data)):
            if self.data.iloc[i][human_actions] == self.data.iloc[i][agent_actions] == 'buy':
                match += 1
                total += 1
            elif self.data.iloc[i][human_actions] == self.data.iloc[i][agent_actions] == 'sell':
                match += 1
                total += 1
            elif self.data.iloc[i][human_actions] != 'None' and self.data.iloc[i][agent_actions] != 'None':
                total += 1
        return match / total
