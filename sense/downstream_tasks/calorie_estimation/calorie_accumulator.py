import numpy as np
import time

from sense.downstream_tasks.postprocess import PostProcessor


class CalorieAccumulator(PostProcessor):
    """
    This class converts MET values into calories and keep tracks of the total number of burned
    calories so far.

    More specifically, met values are first corrected based on the user's personal information
    (see `correct_met_value`) before being smoothed (see `update_running_met_value`) and
    accumulated (see `accumulate`).
    """

    def __init__(self, weight=70, height=170, age=30, gender='unknown', smoothing=20,
                 recovery_factor=60, **kwargs):
        """
        :param weight:           User's weight (in kg).
        :param height:           User's height (in cm).
        :param age:              User's age (in years).
        :param gender:           User's gender ('male', 'female' or other).
        :param smoothing:        Amount of smoothing (in seconds) applied to the input stream of
                                 MET values.
        :param recovery_factor:  Constant that controls how long it takes to return to a
                                 resting MET value. recovery_factor=30 means that it will
                                 take ~2 minutes to get back to a MET value of 1 after
                                 reaching a MET value of 8.
        """
        super().__init__(**kwargs)
        self.weight = weight
        self.height = height
        self.age = age
        self.gender = gender
        self.smoothing = smoothing
        self.recovery_factor = recovery_factor
        self.met_value_running = 0.
        self.calorie_count = 0
        self.buffer = [(5, 0)]  # initialize with 5 seconds of MET=0
        self.time_last_update = None
        self.met_value_live = 0.

    def postprocess(self, met_value_live):
        """
        Converts provided met value to calories and adds it to the total count.
        """
        if met_value_live is not None:
            met_value_live = met_value_live.mean()
            now = time.perf_counter()
            duration = now - (self.time_last_update or now - 1.)
            self.time_last_update = now
            self.buffer.insert(0, (duration, self.correct_met_value(self.met_value_live)))
            self.update_running_met_value(duration)
            self.calorie_count += self.weight * (duration / 3600) * self.met_value_running
            self.met_value_live += 0.2 * (met_value_live - self.met_value_live)
        return {'Total calories': self.calorie_count,
                'Met value': self.met_value_live,
                'Corrected met value': self.met_value_running}

    def update_running_met_value(self, duration):
        """
        Updates the internal running MET value.
        """
        met_value_smoothed = self.average_last_n_seconds_of_met_values()
        if met_value_smoothed > self.met_value_running:
            self.met_value_running = met_value_smoothed
        else:
            # Exponential decay in case the current met value (smoothed) decreased
            # This is to model that we continue burning many calories shortly after exercising
            # The recovery factor should maybe be tuned based on user's fitness level
            self.met_value_running *= np.exp(-duration / self.recovery_factor)

    def average_last_n_seconds_of_met_values(self):
        """
        Returns the average met value over the last `self.smoothing` seconds.
        """
        met_value_avg = 0
        time_window = 0

        # Get last N seconds of MET values
        for idx, (duration, met_value) in enumerate(self.buffer):
            duration = duration - max(time_window + duration - self.smoothing, 0)
            met_value_avg = (time_window * met_value_avg + met_value * duration) / (time_window + duration)
            time_window += duration
            if time_window >= self.smoothing:
                self.buffer = self.buffer[0:idx + 1]    # remove outdated data
                break

        return met_value_avg

    def correct_met_value(self, met_value):
        """
        Predicted met values assume 3.5 ml.kg-1.min-1 as a proxy value for the resting
        metabolic rate (RMR) of 1 MET. Studies have showed that this assumption lead to
        underestimated MET value: see https://sites.google.com/site/compendiumofphysicalactivities/corrected-mets
        """
        rmr = self.RMR * 1000 / (1440 * 5 * self.weight)  # convert kcal/day RMR to ml.kg-1.min-1
        correction_factor = 3.5 / rmr
        return correction_factor * met_value

    @property
    def RMR(self):
        """
        Computes the resting metabolic rate (kcal/day) using the Harris-Benedict equation.

        Sources:
        - Harris JA, Benedict FG (1918). "A Biometric Study of Human Basal Metabolism". Proceedings
        of the National Academy of Sciences of the United States of America.
        4 (12): 370–3. doi:10.1073/pnas.4.12.370. PMC 1091498. PMID 16576330
        - Mifflin MD, St Jeor ST, Hill LA, Scott BJ, Daugherty SA, Koh YO (1990). "A new predictive
        equation for resting energy expenditure in healthy individuals". The American Journal
        of Clinical Nutrition. 51 (2): 241–7. doi:10.1093/ajcn/51.2.241. PMID 2305711
        """
        if self.gender == 'male':
            offset = 5
        elif self.gender == 'female':
            offset = -161
        else:
            # Take the average
            offset = (5 - 161) / 2
        return 10 * self.weight + 6.25 * self.height - 5 * self.age + offset
