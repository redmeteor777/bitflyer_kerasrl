import sys
import gym
import numpy as np
import gym.spaces
import pandas as pd
from logging import getLogger, StreamHandler, FileHandler, DEBUG, INFO
from datetime import datetime

class MyEnv(gym.Env):

    def __init__(self, alpha=.2, epsilon=.1, gamma=.99, actions=None, observation=None):
        super().__init__()

        """calc Settings"""
        self.settings_ = {
            "size": "fix"
        }
        self.logger_settings()

        # action_space, observation_space, reward_range を設定する
        self.action_space = gym.spaces.Discrete(3)  # 売・待・買
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(64,)
        )
        self.reward_range = [-1., 1.]
        self.env_data = pd.read_csv(
            "to_csv_out_sample_1.csv",
            dtype=float,
            )
        self.episode_count = 0
        self.episode_length = len(self.env_data) - 1
        self._reset()
    
    def _reset(self):
        # 状態を初期化し、初期の観測値を返す
        self.flag_initializer()
        self.episode_count += 1
        self.env_data = pd.read_csv(
            "to_csv_out_sample_1.csv",
            header=0,
            dtype=float,
            )
        return self._observe()

    def flag_initializer(self):
        # ループ開始時のフラグ初期化
        self.log = []
        self.entry_signal = None
        self.mode = None
        self.flag_ = {
            "size": 0,
            "side": None,
            "pos_price": None,
            "stop": None,
            "stop_range": None,
            "farest_stop": None,
            "limit": None,
            "collateral": 0,
        }
        self.counter = {
            "entry_buy": 0,
            "entry_sell": 0,
            "win_buy":0,
            "win_sell":0,
        }
        self.reward_ = 0
        self.penalty = False
        self.step_ = 0
        self.done = False

    def flag_cleaner(self):
        # 取引決済後のフラグ初期化
        self.flag_["side"] = None
        self.flag_["pos_price"] = None
        self.flag_["stop"] = None
        self.flag_["stop_range"] = None
        self.flag_["farest_stop"] = None
        self.flag_["limit"] = None
        self.flag_["size"] = 0

    def _step(self, action):
        # action を実行し、結果を返す
        self.tmp_price = self.env_data.loc[self.step_]
        self.next_price = self.env_data.loc[self.step_+1]
        # エントリー判断
        self.judge_entry(action)
        # ストップ幅計算
        self.calc_stop_range()
        # ロット計算
        self.calc_size()
        # エントリー実行
        self.create_entry()
        # ストップ／リミット処理
        self.stop_limit_checker()
        # ポジション決済
        self.position_closer()

        if self.reward_ > 0:
            reward = self.reward_ * 100
        elif self.reward_ < 0:
            reward = self.reward_ * 100
        elif action == 2:
            # reward = -1
            reward = 0
        else:
            reward = self.reward_
            # reward = self.reward_
        self.reward_ = 0
        self.step_ += 1
        if self.step_ == len(self.env_data)-1:
            self.done = True
            print("")
            print("COLLATERAL: {} | BUY: {} / {} | SELL: {} / {}".format(
                self.flag_["collateral"],
                self.counter["win_buy"],
                self.counter["entry_buy"],
                self.counter["win_sell"],
                self.counter["entry_sell"]))
        observation = self._observe()

        return observation, reward, self.done, {}

    def cs(self, val):
        # 正規化された値を復元
        result = int((val * (1495430 - 441900)) + 441900)
        return result

    def stop_limit_checker(self):
        if self.flag_["side"] == "BUY":
            # ロング時の利確損切確認
            if self.flag_["stop"] > self.next_price["low_price"]:
                self.flag_["collateral"] += \
                    (self.cs(self.flag_["pos_price"]) * -1 + self.cs(self.next_price["low_price"])) * self.flag_["size"]
                self.reward_ = (self.flag_["pos_price"] * -1 + self.next_price["low_price"]) * self.flag_["size"]
                self.logger.debug("Side: {} / Size: {} / Price: {} / Close: {} / Collateral: {}".format(
                    self.flag_["side"], self.flag_["size"], self.flag_["pos_price"], self.flag_["limit"], self.flag_["collateral"]))
                self.flag_cleaner()
            if not self.flag_["limit"] is None:
                if self.flag_["limit"] < self.next_price["high_price"]:
                    self.flag_["collateral"] += \
                        (self.cs(self.flag_["pos_price"]) * -1 + self.cs(self.flag_["limit"])) * self.flag_["size"]
                    self.reward_ = (self.flag_["pos_price"] * -1 + self.flag_["limit"]) * self.flag_["size"]
                    self.logger.debug("Side: {} / Size: {} / Price: {} / Close: {} / Collateral: {}".format(
                        self.flag_["side"], self.flag_["size"], self.flag_["pos_price"], self.flag_["stop"], self.flag_["collateral"]))
                    self.flag_cleaner()
                    self.counter["win_buy"] += 1

        if self.flag_["side"] == "SELL":
        # ショート時の利確損切確認
            if self.flag_["stop"] < self.next_price["high_price"]:
                self.flag_["collateral"] += \
                    (self.cs(self.flag_["pos_price"]) - self.cs(self.flag_["stop"])) * self.flag_["size"]
                self.reward_ = (self.flag_["pos_price"] - self.flag_["stop"]) * self.flag_["size"]
                self.logger.debug("Side: {} / Size: {} / Price: {} / Close: {} / Collateral: {}".format(
                    self.flag_["side"], self.flag_["size"], self.flag_["pos_price"], self.flag_["stop"], self.flag_["collateral"]))
                self.flag_cleaner()
            if not self.flag_["limit"] is None:
                if self.flag_["limit"] > self.next_price["low_price"]:
                    self.flag_["collateral"] += \
                        (self.cs(self.flag_["pos_price"]) - self.cs(self.flag_["limit"])) * self.flag_["size"]
                    self.reward_ = (self.flag_["pos_price"] - self.flag_["limit"]) * self.flag_["size"]
                    self.logger.debug("Side: {} / Size: {} / Price: {} / Close: {} / Collateral: {}".format(
                        self.flag_["side"], self.flag_["size"], self.flag_["pos_price"], self.flag_["limit"], self.flag_["collateral"]))
                    self.flag_cleaner()
                    self.counter["win_sell"] += 1

    def judge_entry(self, action):
        # エントリー可否を判断
        if action == 0:
            self.entry_signal = "BUY"
        elif action == 1:
            self.entry_signal = "SELL"

    def position_closer(self):
        # ポジションを決済する
        if self.flag_["side"] == "BUY":
            # ロング時の決済注文
            self.flag_["collateral"] += \
                (self.cs(self.flag_["pos_price"]) * -1 + self.cs(self.next_price["close_price"])) * self.flag_["size"]
            self.reward_ = (self.flag_["pos_price"] * -1 + self.next_price["close_price"]) * self.flag_["size"]
            self.logger.debug("Side: {} / Size: {} / Price: {} / Close: {} / Collateral: {}".format(
                self.flag_["side"], self.flag_["size"], self.flag_["pos_price"], self.next_price["close_price"], self.flag_["collateral"]))
            if self.flag_["pos_price"] < self.next_price["close_price"]:
                self.counter["win_buy"] += 1
            self.flag_cleaner()
        if self.flag_["side"] == "SELL":
            # ショート時の決済注文
            self.flag_["collateral"] += \
                (self.cs(self.flag_["pos_price"]) - self.cs(self.next_price["close_price"])) * self.flag_["size"]
            self.reward_ = (self.flag_["pos_price"] - self.next_price["close_price"]) * self.flag_["size"]
            self.logger.debug("Side: {} / Size: {} / Price: {} / Close: {} / Collateral: {}".format(
                self.flag_["side"], self.flag_["size"], self.flag_["pos_price"], self.next_price["close_price"], self.flag_["collateral"]))
            if self.flag_["pos_price"] > self.next_price["close_price"]:
                self.counter["win_sell"] += 1
            self.flag_cleaner()

    def calc_stop_range(self):
        # ストップ幅の計算
        limit_range = self.tmp_price["min_atr_30"]
        if self.entry_signal == "BUY":        
            self.flag_["stop"] = self.tmp_price["close_price"] - limit_range * 2.5
        elif self.entry_signal == "SELL":
            self.flag_["stop"] = self.tmp_price["close_price"] + limit_range * 2.5

    def calc_size(self):
        # ポジションサイズの計算
        if self.settings_["size"] == "fix":
            self.flag_["size"] = 1
        else:
            self.flag_["size"] = round(self.flag_["collateral"] // 10000 * 0.01, 2)

    def create_entry(self):
        # 注文フラッグを作成
        if self.entry_signal == "BUY":
            self.counter["entry_buy"] += 1
            self.flag_["side"] = "BUY"
            self.flag_["pos_price"] = self.tmp_price["close_price"]
        elif self.entry_signal == "SELL":
            self.counter["entry_sell"] += 1
            self.flag_["side"] = "SELL"
            self.flag_["pos_price"] = self.tmp_price["close_price"]
        self.entry_signal = None

    def _render(self, mode='human', close=False):
        # 環境を可視化する
        pass
    def _close(self):
        # 環境を閉じて後処理をする
        pass
    def _seed(self, seed=None):
        # ランダムシードを固定する
        pass

    def _observe(self):
        if self.done:
            result = self.env_data.loc[self.step_]
        else:
            result = self.env_data.loc[self.step_+1]
        return result

    def logger_settings(self):

        """Logger Settings"""
        self.logger = getLogger(__name__)
        self.logger.setLevel(INFO)
        self.handlerSh = StreamHandler()
        self.handlerSh.setLevel(INFO)
        self.logger.addHandler(self.handlerSh)
