import sys
import numpy as np
from scipy.integrate import odeint
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QGridLayout, QLabel,
    QLineEdit, QPushButton, QProgressBar, QDialog
)
from PyQt6.QtCore import QTimer, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# default parameters
default_params = {
    "a": 0.2,
    "b": 0.2,
    "c": 5.7,
    "t_end": 100,
    "t_points": 1000,
    "split_ratio": 0.333,
    "initial_x": 0.0,
    "initial_y": 1.0,
    "initial_z": 0.0,
    "epochs": 100,
    "batch_size": 1000,
    "lstm_dim": 512,
    "keep_rate": 0.99,
    "learning_rate": 0.001
}

# rossler system function
def rossler(state, t, a, b, c):
    x, y, z = state
    dxdt = -y - z
    dydt = x + a * y
    dzdt = b + z * (x - c)
    return [dxdt, dydt, dzdt]

# worker thread for running the simulation
class SimulationThread(QThread):
    progress_signal = pyqtSignal(int)
    result_signal = pyqtSignal(object)

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.stopped = False

    def stop(self):
        self.stopped = True

    def run(self):
        try:
            # extract parameters
            t = np.linspace(0, self.params["t_end"], self.params["t_points"])
            initial_conditions = [self.params["initial_x"], self.params["initial_y"], self.params["initial_z"]]
            solution = odeint(rossler, initial_conditions, t, args=(self.params["a"], self.params["b"], self.params["c"]))

            # split data
            solution_x = solution[:, 0].reshape(-1, 1)
            solution_y = solution[:, 1].reshape(-1, 1)
            solution_z = solution[:, 2].reshape(-1, 1)
            points_length = int(len(solution) * self.params["split_ratio"])
            input_solution_x = solution_x[:points_length]
            ground_truth_solution_x = solution_x[points_length:]
            input_solution_y = solution_y[:points_length]
            ground_truth_solution_y = solution_y[points_length:]
            input_solution_z = solution_z[:points_length]
            ground_truth_solution_z = solution_z[points_length:]

            if self.stopped:
                return

            # scale data
            scaler_x = MinMaxScaler()
            scaler_y = MinMaxScaler()
            scaler_z = MinMaxScaler()
            X_train_normalized_x = scaler_x.fit_transform(input_solution_x)
            y_train_normalized_x = scaler_x.transform(ground_truth_solution_x)
            X_train_normalized_x = np.reshape(X_train_normalized_x, (-1, 1, 1))
            y_train_normalized_x = np.reshape(y_train_normalized_x, (-1, 1, 1))

            X_train_normalized_y = scaler_y.fit_transform(input_solution_y)
            y_train_normalized_y = scaler_y.transform(ground_truth_solution_y)
            X_train_normalized_y = np.reshape(X_train_normalized_y, (-1, 1, 1))
            y_train_normalized_y = np.reshape(y_train_normalized_y, (-1, 1, 1))

            X_train_normalized_z = scaler_z.fit_transform(input_solution_z)
            y_train_normalized_z = scaler_z.transform(ground_truth_solution_z)
            X_train_normalized_z = np.reshape(X_train_normalized_z, (-1, 1, 1))
            y_train_normalized_z = np.reshape(y_train_normalized_z, (-1, 1, 1))

            if self.stopped:
                return

            # build lstm models
            def build_lstm_model(input_shape):
                model = Sequential([
                    LSTM(self.params["lstm_dim"], input_shape=input_shape, return_sequences=True),
                    Dropout(1 - self.params["keep_rate"]),
                    Bidirectional(LSTM(self.params["lstm_dim"], return_sequences=True)),
                    Dropout(1 - self.params["keep_rate"]),
                    Bidirectional(LSTM(self.params["lstm_dim"])),
                    Dropout(1 - self.params["keep_rate"]),
                    Dense(1)
                ])
                return model

            model_x = build_lstm_model((X_train_normalized_x.shape[1], 1))
            model_y = build_lstm_model((X_train_normalized_y.shape[1], 1))
            model_z = build_lstm_model((X_train_normalized_z.shape[1], 1))

            # compile models with separate optimizer instances
            optimizer_x = Adam(learning_rate=self.params["learning_rate"])
            optimizer_y = Adam(learning_rate=self.params["learning_rate"])
            optimizer_z = Adam(learning_rate=self.params["learning_rate"])

            model_x.compile(optimizer=optimizer_x, loss='mse')
            model_y.compile(optimizer=optimizer_y, loss='mse')
            model_z.compile(optimizer=optimizer_z, loss='mse')

            # train models
            if not self.stopped:
                model_x.fit(X_train_normalized_x, X_train_normalized_x, epochs=self.params["epochs"],
                            batch_size=self.params["batch_size"], verbose=0)
                self.progress_signal.emit(33)

            if not self.stopped:
                model_y.fit(X_train_normalized_y, X_train_normalized_y, epochs=self.params["epochs"],
                            batch_size=self.params["batch_size"], verbose=0)
                self.progress_signal.emit(66)

            if not self.stopped:
                model_z.fit(X_train_normalized_z, X_train_normalized_z, epochs=self.params["epochs"],
                            batch_size=self.params["batch_size"], verbose=0)
                self.progress_signal.emit(100)

            # predictions
            if self.stopped:
                return

            predicted_solution_x = scaler_x.inverse_transform(model_x.predict(y_train_normalized_x)).flatten()
            predicted_solution_y = scaler_y.inverse_transform(model_y.predict(y_train_normalized_y)).flatten()
            predicted_solution_z = scaler_z.inverse_transform(model_z.predict(y_train_normalized_z)).flatten()

            # mse calculations
            mse_x = np.mean((ground_truth_solution_x.flatten() - predicted_solution_x) ** 2)
            mse_y = np.mean((ground_truth_solution_y.flatten() - predicted_solution_y) ** 2)
            mse_z = np.mean((ground_truth_solution_z.flatten() - predicted_solution_z) ** 2)
            combined_mse = (mse_x + mse_y + mse_z) / 3

            # return results
            self.result_signal.emit({
                "t": t,
                "input_x": input_solution_x,
                "input_y": input_solution_y,
                "input_z": input_solution_z,
                "ground_truth_x": ground_truth_solution_x,
                "ground_truth_y": ground_truth_solution_y,
                "ground_truth_z": ground_truth_solution_z,
                "predicted_x": predicted_solution_x,
                "predicted_y": predicted_solution_y,
                "predicted_z": predicted_solution_z,
                "mse_x": mse_x,
                "mse_y": mse_y,
                "mse_z": mse_z,
                "combined_mse": combined_mse
            })
        except Exception as e:
            self.result_signal.emit({"error": str(e)})

# main application window
class RosslerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rossler Attractor Simulation")
        self.setGeometry(100, 100, 1200, 800)
        self.simulation_thread = None
        self.init_ui()

    def init_ui(self):
        # main widget
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # parameter grid
        param_grid = QGridLayout()
        self.param_inputs = {}
        for i, (param, value) in enumerate(default_params.items()):
            label = QLabel(f"{param}:")
            input_field = QLineEdit(str(value))
            self.param_inputs[param] = input_field
            param_grid.addWidget(label, i, 0)
            param_grid.addWidget(input_field, i, 1)
        layout.addLayout(param_grid)

        # run and stop buttons
        button_layout = QVBoxLayout()
        self.run_button = QPushButton("Run Simulation")
        self.run_button.clicked.connect(self.run_simulation)
        button_layout.addWidget(self.run_button)

        self.stop_button = QPushButton("Stop Simulation")
        self.stop_button.clicked.connect(self.stop_simulation)
        self.stop_button.setDisabled(True)
        button_layout.addWidget(self.stop_button)
        layout.addLayout(button_layout)

        # canvas for plots
        self.canvas = FigureCanvas(plt.figure(figsize=(14, 10)))
        layout.addWidget(self.canvas)

    def run_simulation(self):
        # disable run button and enable stop button
        self.run_button.setDisabled(True)
        self.stop_button.setDisabled(False)

        # read parameters
        params = {}
        for param, input_field in self.param_inputs.items():
            value = input_field.text()
            if param in ["t_points", "epochs", "batch_size", "lstm_dim"]:
                params[param] = int(value)
            elif param in ["split_ratio", "keep_rate", "learning_rate", "a", "b", "c"]:
                params[param] = float(value)
            else:
                params[param] = float(value)

        # loading dialog
        loading_dialog = QDialog(self)
        loading_dialog.setWindowTitle("Generating Predictions")
        layout = QVBoxLayout(loading_dialog)
        self.progress_bar = QProgressBar()
        layout.addWidget(QLabel("Generating predictions, please wait..."))
        layout.addWidget(self.progress_bar)
        timer = QTimer(self)
        timer.timeout.connect(self.animate_progress_bar)
        timer.start(50)
        loading_dialog.show()

        # run simulation thread
        self.simulation_thread = SimulationThread(params)
        self.simulation_thread.progress_signal.connect(self.progress_bar.setValue)
        self.simulation_thread.result_signal.connect(lambda results: self.update_plot(results, loading_dialog, timer))
        self.simulation_thread.start()

    def stop_simulation(self):
        if self.simulation_thread:
            self.simulation_thread.stop()
            self.simulation_thread.terminate()  # Ensure thread is stopped
            self.simulation_thread = None
            self.run_button.setDisabled(False)
            self.stop_button.setDisabled(True)

    def animate_progress_bar(self):
        value = self.progress_bar.value()
        self.progress_bar.setValue((value + 5) % 105)

    def update_plot(self, results, loading_dialog, timer):
        timer.stop()
        loading_dialog.close()
        self.run_button.setDisabled(False)
        self.stop_button.setDisabled(True)

        if "error" in results:
            error_dialog = QDialog(self)
            error_dialog.setWindowTitle("Error")
            error_label = QLabel(results["error"], error_dialog)
            error_label.move(10, 10)
            error_dialog.exec()
            return

        # update plots (same logic as before)
        t = results["t"]
        fig = self.canvas.figure
        fig.clf()
        grid = plt.GridSpec(3, 4, hspace=0.4, wspace=0.3)

        # 2d plots with mse
        ax_x = fig.add_subplot(grid[0, :2])
        ax_x.plot(t[:len(results["input_x"])], results["input_x"], label='Input', color='black')
        ax_x.plot(t[len(results["input_x"]):], results["ground_truth_x"], label='Ground Truth', color='orange')
        ax_x.plot(t[len(results["input_x"]):], results["predicted_x"], label='Prediction', linestyle='--', color='blue')
        ax_x.set_title(f"X-Axis (MSE: {results['mse_x']:.5f})")
        ax_x.legend()

        ax_y = fig.add_subplot(grid[1, :2])
        ax_y.plot(t[:len(results["input_y"])], results["input_y"], label='Input', color='black')
        ax_y.plot(t[len(results["input_y"]):], results["ground_truth_y"], label='Ground Truth', color='orange')
        ax_y.plot(t[len(results["input_y"]):], results["predicted_y"], label='Prediction', linestyle='--', color='blue')
        ax_y.set_title(f"Y-Axis (MSE: {results['mse_y']:.5f})")
        ax_y.legend()

        ax_z = fig.add_subplot(grid[2, :2])
        ax_z.plot(t[:len(results["input_z"])], results["input_z"], label='Input', color='black')
        ax_z.plot(t[len(results["input_z"]):], results["ground_truth_z"], label='Ground Truth', color='orange')
        ax_z.plot(t[len(results["input_z"]):], results["predicted_z"], label='Prediction', linestyle='--', color='blue')
        ax_z.set_title(f"Z-Axis (MSE: {results['mse_z']:.5f})")
        ax_z.legend()

        # 3d plot with combined mse
        ax_3d = fig.add_subplot(grid[:, 2:], projection='3d')
        ax_3d.scatter(results["input_x"].flatten(), results["input_y"].flatten(), results["input_z"].flatten(),
                      label='Input', color='black', s=20)
        ax_3d.plot(results["ground_truth_x"].flatten(), results["ground_truth_y"].flatten(),
                   results["ground_truth_z"].flatten(), label='Ground Truth', color='orange')
        ax_3d.scatter(results["predicted_x"], results["predicted_y"], results["predicted_z"],
                      label='Prediction', color='blue')
        ax_3d.set_title(f"3D Plot (Combined MSE: {results['combined_mse']:.5f})")
        ax_3d.legend()

        self.canvas.draw()

# run application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RosslerApp()
    window.show()
    sys.exit(app.exec())


