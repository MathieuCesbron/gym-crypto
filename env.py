import numpy as np
import gym
import random
from gym import spaces
import static


class CryptoEnv(gym.Env):
    def __init__(self, df):
        self.df = df
        self.reward_range = (0, static.MAX_ACCOUNT_BALANCE)
        self.total_fees = 0
        self.total_volume_traded = 0
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
                                            shape=(10, 5),
                                            dtype=np.float16)

    def reset(self):
        self.balance = static.INITIAL_ACCOUNT_BALANCE
        self.net_worth = static.INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = static.INITIAL_ACCOUNT_BALANCE
        self.total_fees = 0
        self.total_volume_traded = 0
        self.crypto_held = 0
        self.current_step = 4

        return self._next_observation()

    def _next_observation(self):
        #Get the data for the last 5 timestep
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

        # We will Append additional data to render after
        obs = np.append(frame, [[
            self.balance / static.MAX_ACCOUNT_BALANCE, self.net_worth /
            self.max_net_worth, self.crypto_held / static.MAX_CRYPTO, 0, 0
        ]],
                        axis=0)

        return obs

    def _take_action(self, action):  #pylint: disable=method-hidden
        # Set the current price to a random price between open and close
        current_price = random.uniform(
            self.df.loc[self.current_step, 'Real open'],
            self.df.loc[self.current_step, 'Real close'])

        if action[0] > 0:
            # Buy
            crypto_bought = self.balance * action[0] / current_price
            self.total_fees += crypto_bought * current_price * static.MAKER_FEE
            self.total_volume_traded += crypto_bought * current_price
            self.balance -= crypto_bought * current_price
            self.crypto_held += crypto_bought

        if action[0] < 0:
            # Sell
            crypto_sold = -self.crypto_held * action[0]
            self.total_fees += crypto_sold * current_price * static.TAKER_FEE
            self.total_volume_traded += crypto_sold * current_price
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
        if self.current_step > len(self.df.loc[:, 'Open'].values) - 1:
            self.current_step = 0

        delay_modifier = self.current_step / static.MAX_STEPS

        # Is it net_worth or balance ?
        reward = self.net_worth * delay_modifier
        done = self.net_worth <= 0

        obs = self._next_observation()

        # {} needed because gym wants 4 args
        return obs, reward, done, {}

    def render(self):
        profit = self.net_worth - static.INITIAL_ACCOUNT_BALANCE

        print("----------------------------------------")
        print("Step: " + str(self.current_step))
        print("Balance: " + str(self.balance))
        print("Crypto held: " + str(self.crypto_held))
        print("Fees paid: " + str(self.total_fees))
        print("Volume traded: " + str(self.total_volume_traded))
        print("Net worth: " + str(self.net_worth))
        print("Max net worth: " + str(self.max_net_worth))
        print("Profit: " + str(profit))
