from ray.tune import Callback
import pandas as pd


class AvgMetricCallback(Callback):
    def __init__(self):
        super(AvgMetricCallback, self).__init__()

    def init(self):
        try:
            self.results_df
            self.record_index += 1
        except:
            self.results_df = pd.DataFrame()
            self.record_index = 1

    def handle_parameters(self, config):
        for key, value in config.items():
            if isinstance(value, list):
                config[key] = str(value)

        df = pd.DataFrame(config, index=[self.record_index])
        return df

    def on_trial_complete(self, iteration, trials, trial, **info):
        self.init()

        config_df = self.handle_parameters(trial.config)
        config_df['trial'] = trial
        self.results_df = pd.concat([self.results_df, config_df], sort=False)