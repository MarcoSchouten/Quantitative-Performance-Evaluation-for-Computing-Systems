import pickle
import numpy as np


class EpochOptimizer:

    def __init__(self, cores, paralellization, batchsize, lr):
        self._cores = cores
        self._parallelization = paralellization
        self._batch_size = batchsize
        self._learning_rate = lr
        self._model = self.__load_model()

    def __load_model(self):
        # Loads the stored RF model
        try:
            model = pickle.load(open("./models/model.pkl", "rb"))
        except:
            print('Failed to load model')
            model = None

        return model

    def __doe_reg(self, epoch):
        # Regression Model derived from DoE

        duration = 136.9 - 38.8 * self._cores - 49.9 * self._parallelization - 0.499 * self._batch_size \
                   - 1335 * self._learning_rate + 115.60 * epoch + 21.18 * self._cores * self._parallelization \
                   + 0.1077 * self._cores * self._batch_size + 222 * self._cores * self._learning_rate \
                   - 24.01 * self._cores * epoch + 0.1613 * self._parallelization * self._batch_size \
                   + 502 * self._parallelization * self._learning_rate - 15.912 * self._parallelization * epoch \
                   + 3.60 * self._batch_size * self._learning_rate - 0.06528 * self._batch_size * epoch \
                   - 47.3 * self._learning_rate * epoch \
                   - 0.0550 * self._cores * self._parallelization * self._batch_size \
                   - 212.8 * self._cores * self._parallelization * self._learning_rate \
                   + 2.495 * self._cores * self._parallelization * epoch \
                   + 0.02080 * self._cores * self._batch_size * epoch \
                   + 45.2 * self._cores * self._learning_rate * epoch \
                   - 0.401 * self._batch_size * self._learning_rate * epoch

        return duration

    def __rf_reg(self, epoch):
        # Prediction from RF model
        if self._model is None:
            return None

        x_pred = np.array([[self._cores, self._parallelization, self._batch_size, self._learning_rate, epoch]])
        y_pred = self._model.predict(x_pred)

        return y_pred[0]

    def ensemble(self, epoch):
        # Creates the response based on the two predictions
        dur_doe = self.__doe_reg(epoch)

        dur_rf = self.__rf_reg(epoch)

        duration = None

        if dur_rf is not None and dur_doe > 0:
            duration = (0.6 * dur_doe + 0.4 * dur_rf)
        elif dur_doe < 0:
            duration = dur_rf

        return round(duration)

    def optimize(self, lowerbound, upperbound, timelimit, steps=5):
        # Optimiser function

        assert lowerbound > 0, "Lower Bound should be greater than 0"
        assert upperbound > 0, "Upper Bound should be greater than 0"
        assert upperbound > lowerbound, "Upper Bound should be greater than Lower Bound"

        if self.ensemble(upperbound) <= timelimit:
            return upperbound

        op_epoch = lowerbound
        curr_duration = self.ensemble(lowerbound)
        if curr_duration > timelimit:
            print('No value found within bound')
            return None
        else:
            for epoch in range(lowerbound, upperbound + steps, steps):

                curr_duration = self.ensemble(epoch)
                if curr_duration <= timelimit:
                    op_epoch = epoch
                else:
                    break

        return op_epoch


if __name__ == '__main__':
    CONFIGS = [[3, 1, 0.01, 512],
               [3, 3, 0.05, 512],
               [1, 3, 0.01, 512],
               [1, 3, 0.05, 256],
               [3, 1, 0.05, 256],
               [3, 3, 0.01, 256]]

    for item in CONFIGS:
        (EXECUTOR_CORES, DATA_PARALLEL, LEARNING_RATE, BATCH_SIZE) = item
        eo = EpochOptimizer(cores=EXECUTOR_CORES, paralellization=DATA_PARALLEL, batchsize=BATCH_SIZE, lr=LEARNING_RATE)
        optimum_max_epoch = eo.optimize(1, 20, 250, 1)
        print(optimum_max_epoch)  # 8 17 9 6 6 12
