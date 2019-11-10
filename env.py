import numpy as np
import gym
import random
from gym import spaces
from static import *


class CryptoEnv(gym.Env):
    def __init__(self, df):
        self.df = df
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)
        # Could be remove when more data will be added
        self.crypto_held = 0
        # Action space from -1 to 1, -1 is short, 1 is buy
        self.action_space = spaces.Box(low=-1,
                                       high=1,
                                       shape=(1, ),
                                       dtype=np.float16)
        # Observation space contains only the actual price for the moment
        self.observation_space = spaces.Box(low=0,
                                            high=1,
                                            shape=(1, ),
                                            dtype=np.float16)

    def reset(self):
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.current_step = 0

        return self._next_observation()

    def _next_observation(self):
        #Get the actual price and scale it
        frame = np.array(
            [self.df.loc[self.current_step, 'open'] / MAX_BTC_PRICE])

        # We will Append additional data to render after
        obs = np.append(frame, "add_data")

        return frame

    def _take_action(self, action):  #pylint: disable=method-hidden
        # Set the current price to a random price between open and close
        current_price = random.uniform(self.df.loc[self.current_step, 'open'],
                                       self.df.loc[self.current_step, 'close'])

        if action > 0:
            # Buy
            crypto_bought = self.balance * action / current_price
            self.balance -= crypto_bought * current_price
            self.crypto_held += crypto_bought

        if action < 0:
            # Sell
            crypto_sold = -self.crypto_held * action
            self.balance += crypto_sold * current_price
            self.crypto_held -= crypto_sold

        self.net_worth = self.balance + self.crypto_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        # Will be updated later (may be remove -1)
        if self.current_step > len(self.df.loc[:, 'open'].values) - 1:
            self.current_step = 0

        delay_modifier = self.current_step / MAX_STEPS

        # Is it net_worth or balance ?
        reward = self.net_worth * delay_modifier
        done = self.net_worth <= 0

        obs = self._next_observation()

        # {} needed because gym wants 4 args
        return obs, reward, done, {}

    def render(self):
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        print("Step: " + str(self.current_step))
        print("Balance: " + str(self.balance))
        print("Crypto held: " + str(self.crypto_held))
        print("Net worth: " + str(self.net_worth))
        print("Profit: " + str(profit))
