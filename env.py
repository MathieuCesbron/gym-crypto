import numpy as np
import gym
import random
from gym import spaces
import static

import matplotlib.pyplot as plt
import matplotlib


class CryptoEnv(gym.Env):
    def __init__(self, df, title=None):
        self.df = df
        self.reward_range = (-static.MAX_ACCOUNT_BALANCE,
                             static.MAX_ACCOUNT_BALANCE)
        self.total_fees = 0
        self.total_volume_traded = 0
        self.crypto_held = 0
        self.bnb_usdt_held = static.BNBUSDTHELD
        self.bnb_usdt_held_start = static.BNBUSDTHELD
        self.episode = 1
        self.graph_to_render = []
        # Action space from -1 to 1, -1 is short, 1 is buy
        self.action_space = spaces.Box(low=-1,
                                       high=1,
                                       shape=(1, ),
                                       dtype=np.float16)
        # Observation space contains only the actual price for the moment
        self.observation_space = spaces.Box(low=0,
                                            high=1,
                                            shape=(10, 5),
                                            dtype=np.float16)

    def reset(self):
        self.balance = static.INITIAL_ACCOUNT_BALANCE
        self.net_worth = static.INITIAL_ACCOUNT_BALANCE + static.BNBUSDTHELD
        self.max_net_worth = static.INITIAL_ACCOUNT_BALANCE + static.BNBUSDTHELD
        self.total_fees = 0
        self.total_volume_traded = 0
        self.crypto_held = 0
        self.bnb_usdt_held = static.BNBUSDTHELD
        self.episode_reward = 0

        # Set the current step to a random point within the data frame
        # Weights of the current step follow the square function
        start = list(range(4, len(self.df.loc[:, 'Open'].values) - static.MAX_STEPS)) + self.df.index[0]
        weights = [i**2 for i in start]
        self.current_step = random.choices(start, weights)[0]
        self.start_step = self.current_step

        return self._next_observation()

    def _next_observation(self):
        # Get the data for the last 5 timestep
        frame = np.array([
            self.df.loc[self.current_step - 4:self.current_step, 'Open'],
            self.df.loc[self.current_step - 4:self.current_step, 'High'],
            self.df.loc[self.current_step - 4:self.current_step, 'Low'],
            self.df.loc[self.current_step - 4:self.current_step, 'Close'],
            self.df.loc[self.current_step - 4:self.current_step, 'Volume'],
            self.df.loc[self.current_step -
                        4:self.current_step, 'Quote asset volume'],
            self.df.loc[self.current_step -
                        4:self.current_step, 'Number of trades'],
            self.df.loc[self.current_step -
                        4:self.current_step, 'Taker buy base asset volume'],
            self.df.loc[self.current_step -
                        4:self.current_step, 'Taker buy quote asset volume']
        ])
        # We append additional data
        obs = np.append(frame, [[self.balance / static.MAX_ACCOUNT_BALANCE,
                                 self.net_worth / self.max_net_worth,
                                 self.crypto_held / static.MAX_CRYPTO,
                                 self.bnb_usdt_held / self.bnb_usdt_held_start,
                                 0]],
                                 axis=0)

        return obs

    def _take_action(self, action):
        # Set the current price to a random price between open and close
        current_price = random.uniform(
            self.df.loc[self.current_step, 'Real open'],
            self.df.loc[self.current_step, 'Real close'])

        if action[0] > 0:
            # Buy
            crypto_bought = self.balance * action[0] / current_price
            self.bnb_usdt_held -= crypto_bought * current_price * static.MAKER_FEE
            self.total_fees += crypto_bought * current_price * static.MAKER_FEE
            self.total_volume_traded += crypto_bought * current_price
            self.balance -= crypto_bought * current_price
            self.crypto_held += crypto_bought

        if action[0] < 0:
            # Sell
            crypto_sold = -self.crypto_held * action[0]
            self.bnb_usdt_held -= crypto_sold * current_price * static.TAKER_FEE
            self.total_fees += crypto_sold * current_price * static.TAKER_FEE
            self.total_volume_traded += crypto_sold * current_price
            self.balance += crypto_sold * current_price
            self.crypto_held -= crypto_sold

        self.net_worth = self.balance + self.crypto_held * current_price + self.bnb_usdt_held

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

    def step(self, action, end=True):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        # Calculus of the reward
        profit = self.net_worth - (static.INITIAL_ACCOUNT_BALANCE +
                                   static.BNBUSDTHELD)
        reward = profit

        # A single episode can last a maximum of MAX_STEPS steps
        if self.current_step >= static.MAX_STEPS + self.start_step:
            end = True
        else:
            end = False

        done = self.net_worth <= 0 or self.bnb_usdt_held <= 0 or end

        if done and end:
            self.episode_reward = reward
            self._render_episode()
            self.graph_to_render.append(reward)
            self.episode += 1

        obs = self._next_observation()

        # {} needed because gym wants 4 args
        return obs, reward, done, {}

    def render(self, print_step=False, graph_reward=False, *args):
        profit = self.net_worth - (static.INITIAL_ACCOUNT_BALANCE +
                                   static.BNBUSDTHELD)

        profit_percent = profit / (static.INITIAL_ACCOUNT_BALANCE +
                                   static.BNBUSDTHELD) * 100

        benchmark_profit = (self.df.loc[self.current_step, 'Real open'] /
                            self.df.loc[self.start_step, 'Real open'] -
                            1) * 100

        if print_step:
            print("----------------------------------------")
            print(f'Step: {self.current_step}')
            print(f'Balance: {round(self.balance, 2)}')
            print(f'Crypto held: {round(self.crypto_held, 2)}')
            print(f'Fees paid: {round(self.total_fees, 2)}')
            print(f'Volume traded: {round(self.total_volume_traded, 2)}')
            print(f'Net worth: {round(self.max_net_worth, 2)}')
            print(f'Max net worth: {round(self.max_net_worth, 2)}')
            print(f'Profit: {round(profit_percent, 2)}% ({round(profit, 2)})')
            print(f'Benchmark profit: {round(benchmark_profit, 2)}')

        # Plot the graph of the reward
        if graph_reward:
            plt.xlabel = ('Episodes')
            plt.ylabel = ('Reward')
            plt.plot(self.graph_to_render)
            plt.savefig('render/graphreward.png')
            plt.show()

        return profit_percent, benchmark_profit

    def _render_episode(self, filename='render/render.txt'):
        file = open(filename, 'a')
        file.write('-----------------------\n')
        file.write(f'Episode numero: {self.episode}\n')
        file.write(f'Profit: {round(self.render()[0], 2)}%\n')
        file.write(f'Benchmark profit: {round(self.render()[1], 2)}%\n')
        file.write(f'Reward: {round(self.episode_reward, 2)}\n')
        file.close()