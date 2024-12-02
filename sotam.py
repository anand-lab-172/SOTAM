import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, explained_variance_score, median_absolute_error
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Input
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Attention
import plotly.graph_objects as go
import plotly.io as pio
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore')

class VLSTM:
    def __init__(self, target, sequence=30, batch_size=48, n1=300, n2=200, d1=25, d2=1, epochs=10, lr=0.001, 
                 dropout=0.0, train_size=0.8, model_name='best_model.keras', loss='mae', activation='linear', early_stopping=False, patience=10):
        """
        Initialize the VLSTM class with the given parameters.

        Parameters:
            target (str): The target feature for prediction.
            sequence_length (int): The length of the input sequences.
            batch_size (int): The batch size for training.
            n1 (int): Number of units in the first LSTM layer.
            n2 (int): Number of units in the second LSTM layer.
            d1 (int): Number of units in the first Dense layer.
            d2 (int): Number of units in the output Dense layer.
            epochs (int): Number of epochs for training.
            learning_rate (float): Learning rate for the optimizer.
            dropout (float): Dropout rate for regularization.
            train_size (float): Proportion of the data to use for training.
            best_model_path (str): Path to save the best model.
            loss (str): Loss function for training.
            activation (str): Activation function for the output layer.
            early_stopping (bool): Whether to use early stopping during training.
            patience (int): Number of epochs to wait for improvement before stopping.
        """
        self.sequence_length = sequence
        self.batch_size = batch_size
        self.n1 = n1
        self.n2 = n2
        self.d1 = d1
        self.d2 = d2
        self.epochs = epochs
        self.learning_rate = lr
        self.Dropout = dropout
        self.train_size = train_size
        self.target = target
        self.model = None
        self.best_model_path = model_name
        self.scalers = {}
        self.activation = activation
        self.history = None
        self.early_stopping = early_stopping
        self.patience = patience

        allowed_losses = ['mae', 'mse', 'mape', 'msle', 'hinge', 'binary_crossentropy', 
                   'categorical_crossentropy', 'sparse_categorical_crossentropy', 
                   'kld', 'poisson', 'cosine_similarity', 'logcosh', 'huber']
        if loss in allowed_losses:
            self.loss = loss
        else:
            raise ValueError(f"Invalid loss function. Allowed values are {allowed_losses}")

    def create_sequences_optimized(self, data):
        """
        Create input sequences and corresponding target values from the data.

        Parameters:
            data (pd.DataFrame): The input data.

        Returns:
            tuple: Input sequences (xs) and target values (ys).
        """
        data_values = data.values.astype('float32')
        # Check if there are enough samples to create sequences
        if len(data) <= self.sequence_length:
            raise ValueError(f"Not enough data to create sequences. Minimum data length should be {self.sequence_length + 1}.")
        
        num_samples = len(data) - self.sequence_length
        num_features = data.shape[1]

        xs = np.empty((num_samples, self.sequence_length, num_features), dtype='float32')
        ys = np.empty(num_samples, dtype='float32')

        for i in range(num_samples):
            xs[i] = data_values[i:i+self.sequence_length]
            ys[i] = data_values[i+self.sequence_length, self.target_idx]

        return xs, ys

    def prepare_data(self, df, features):
        """
        Prepare the data for training and prediction.

        Parameters:
            df (pd.DataFrame): The input DataFrame.
            features (list): List of feature columns to use.

        Returns:
            tuple: Prepared input sequences (X) and target values (y).
        """
        if self.target not in features:
            raise ValueError(f"Target '{self.target}' is not present in the feature list.")
        
        data = df[features].copy()
        for feature in features:
            scaler = MinMaxScaler(feature_range=(0, 1))
            data[feature] = scaler.fit_transform(data[[feature]])
            self.scalers[feature] = scaler

        self.target_idx = features.index(self.target)
        return self.create_sequences_optimized(data)


    def build_model(self, input_shape):
        """
        Build the LSTM model.

        Parameters:
            input_shape (tuple): The shape of the input data.

        Returns:
            Sequential: The compiled LSTM model.
        """
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(LSTM(self.n1, return_sequences=True))
        model.add(LSTM(self.n2, return_sequences=False))
        model.add(Dense(self.d1))
        model.add(Dropout(self.Dropout))
        model.add(Dense(self.d2, activation=self.activation))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=self.loss)
        return model

    def train(self, df, features):
        """
        Train the LSTM model.

        Parameters:
            df (pd.DataFrame): The training data.
            features (list): List of feature columns to use.

        Returns:
            tuple: Training history, actual test values, predicted test values, R² score for train and test sets.
        """
        X, y = self.prepare_data(df, features)

        # Ensure there are enough samples to split
        if len(X) <= self.sequence_length:
            raise ValueError("Not enough data to split into training and testing sets.")
            
        split_ratio = self.train_size
        split = int(split_ratio * len(X))

        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        strategy = tf.distribute.MirroredStrategy()
        print(f"Number of devices: {strategy.num_replicas_in_sync}")

        with strategy.scope():
            self.model = self.build_model((self.sequence_length, X_train.shape[2]))

        callbacks = [ModelCheckpoint(self.best_model_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)]

        if self.early_stopping:
            callbacks.append(EarlyStopping(monitor='val_loss', patience=self.patience, verbose=1))

        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            verbose=1,
            callbacks=callbacks)
                
        self.model = load_model(self.best_model_path)

        y_pred = self.model.predict(X_test)
        y_train_pred = self.model.predict(X_train)

        y_test_actual = self.scalers[self.target].inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_actual = self.scalers[self.target].inverse_transform(y_pred).flatten()
        y_train_actual = self.scalers[self.target].inverse_transform(y_train.reshape(-1, 1)).flatten()
        y_train_pred_actual = self.scalers[self.target].inverse_transform(y_train_pred).flatten()

        train_score = round(r2_score(y_train_actual, y_train_pred_actual)*100,2)
        test_score = round(r2_score(y_test_actual, y_pred_actual)*100,2)

        return self.history, y_test_actual, y_pred_actual, train_score, test_score


    def predict(self, data, features):
        """
        Predict future values using the trained model.

        Parameters:
            data (pd.DataFrame): The input data for prediction.
            features (list): List of feature columns to use.

        Returns:
            np.ndarray: Predicted values.
        """
        data = data[features].copy()
        for feature in features:
            if feature in self.scalers:
                scaler = self.scalers[feature]
                data[feature] = scaler.transform(data[[feature]])
            else:
                raise ValueError(f"Feature '{feature}' not found in scalers. Make sure to use the same features as in training.")

        X, _ = self.create_sequences_optimized(data)
        model = load_model(self.best_model_path)
        y_pred = model.predict(X)
        y_cust_pred = self.scalers[self.target].inverse_transform(y_pred).flatten()
        return y_cust_pred
    
    def forecast(self, data, features, steps, noise_factor=0.02):
        """
        Generate multi-step forecasts with noise using the trained model.

        Parameters:
            data (pd.DataFrame): The input data.
            features (list): List of feature columns to use.
            steps (int): Number of steps to forecast.
            noise_factor (float): Factor to introduce noise in predictions (optional).

        Returns:
            list: A list of forecasted values with added noise.
        """
        # Copy and scale the input data
        scaled_data = data[features].copy()
        for feature in features:
            if feature in self.scalers:
                scaler = self.scalers[feature]
                scaled_data[feature] = scaler.transform(data[[feature]])
            else:
                raise ValueError(f"Feature '{feature}' not found in scalers. Ensure consistent feature usage.")

        # Start with the last available sequence
        sequence = scaled_data.values[-self.sequence_length:].copy()
        forecast = []

        for _ in range(steps):
            input_seq = np.expand_dims(sequence, axis=0)  # Shape: (1, sequence_length, num_features)
            
            # Predict the next step
            predicted = self.model.predict(input_seq, verbose=0)
            predicted_value = predicted.flatten()[0]
            
            # Add noise to simulate realistic fluctuation
            noisy_prediction = predicted_value * (1 + np.random.uniform(-noise_factor, noise_factor))
            
            forecast.append(noisy_prediction)

            # Update the sequence with the new predicted value
            new_row = sequence[-1].copy()  # Take the last row as a base for the next step
            new_row[self.target_idx] = noisy_prediction  # Update the target feature
            sequence = np.vstack([sequence[1:], new_row])  # Shift the window forward

        # Inverse transform the forecasted target values
        forecast_actual = self.scalers[self.target].inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
        return forecast_actual
    

    def plot_forecast(self, data, features, steps, noise_factor=0.2):
        """
        Plot the original data and forecasted data.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing the target feature.
            features (list): List of feature columns to use.
            steps (int): Number of time steps to forecast.
            noise_factor (float): The noise factor to add to the forecasted data.
        """

        original_data = data[self.target].tail(steps).to_list()
        forecasted_data = self.forecast(data, features, steps, noise_factor)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(data[self.target]) - steps, len(data[self.target]))), 
            y=original_data,
            mode='lines+markers',
            name='Original Data',
            line=dict(color='blue')
        ))

        fig.add_trace(go.Scatter(
            x=list(range(len(data[self.target]), len(data[self.target]) + steps)), 
            y=forecasted_data,
            mode='lines+markers',
            name='Forecasted Data',
            line=dict(color='red', dash='dot') 
        ))

        fig.update_layout(
            title='Original Data vs. Forecasted Data',
            xaxis_title='Time',
            yaxis_title='Close Price',
            template='plotly_dark'
        )

        fig.show()


    def evaluate(self, y_actual, y_pred, print_metrics=False):
        """
        Evaluate the model's performance using MAE, RMSE, MAPE, MAD, Explained Variance, Max Error, MSE, and Median Absolute Error.

        Parameters:
            y_actual (np.ndarray): The actual target values.
            y_pred (np.ndarray): The predicted target values.
            print_metrics (bool): If True, print the evaluation metrics. Defaults to False.

        Returns:
            dict: A dictionary containing MAE, RMSE, MAPE, MAD, Explained Variance, Max Error, MSE, and Median Absolute Error scores.
        """
        # Ensure no division by zero in MAPE calculation
        non_zero_indices = y_actual != 0

        # Calculate the metrics
        mae = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        mape = (np.mean(np.abs((y_actual[non_zero_indices] - y_pred[non_zero_indices]) / y_actual[non_zero_indices])) * 100 
                if np.any(non_zero_indices) else np.inf)
        mad = np.mean(np.abs(y_actual - y_pred))
        explained_variance = explained_variance_score(y_actual, y_pred)
        max_error = np.max(np.abs(y_actual - y_pred))
        mse = mean_squared_error(y_actual, y_pred)
        median_absolute_error_value = median_absolute_error(y_actual, y_pred)

        # Print the metrics if print_metrics is True
        if print_metrics:
            print(f"Evaluation Metrics:\n"
                f"Mean Absolute Error (MAE): {mae:.4f}\n"
                f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%\n"
                f"Mean Absolute Deviation (MAD): {mad:.4f}\n"
                f"Root Mean Squared Error (RMSE): {rmse:.4f}\n"
                f"Explained Variance Score: {explained_variance:.4f}\n"
                f"Max Error: {max_error:.4f}\n"
                f"Mean Squared Error (MSE): {mse:.4f}\n"
                f"Median Absolute Error: {median_absolute_error_value:.4f}")

        # Return the metrics as a dictionary
        return {'MAE': mae, 'MAPE': mape, 'MAD': mad, 'MSE': mse, 'RMSE': rmse,
                'Explained Variance': explained_variance, 'Max Error': max_error,
                'Median Absolute Error': median_absolute_error_value}

    
    def summary(self):
        """
        To print the summary of the best model which is saved.
        """
        print(self.model.summary())

    def plot_loss(self, history):
        """
        Plot the training and validation loss over epochs.

        Parameters:
            history: Training history object.
        """
        if not self.history:
            raise ValueError("No training history found. Train the model before plotting.")
        
        fig = go.Figure()

        epochs = list(range(1, len(history.history['loss']) + 1))
        
        fig.add_trace(go.Scatter(
            x=epochs, 
            y=history.history['loss'], 
            mode='lines+markers', 
            name='Train Loss', 
            line=dict(color='cadetblue')))
        
        fig.add_trace(go.Scatter(
            x=epochs,
            y=history.history['val_loss'],
            mode='lines',
            name='Validation Loss',
            line=dict(color='burlywood')))

        fig.update_layout(
            title='Model Loss',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            legend=dict(x=0, y=1.0),
            width=1000,
            height=600,
            xaxis=dict(
                tickmode='array', 
                tickvals=epochs
            ),
            yaxis=dict(gridcolor='lightgrey'),
            plot_bgcolor='white'
        )
        fig.show()

    def prediction_plot(self, y_test, y_pred):
        """
        Plot the actual and predicted data.

        Parameters:
            np.ndarray: Actual test data values.
            np.ndarray: Predicted test data values.
        """
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=list(range(len(y_test))),
            y=y_test.flatten(),
            mode='lines',
            name='Actual'
        ))

        fig.add_trace(go.Scatter(
            x=list(range(len(y_pred))),
            y=y_pred.flatten(),
            mode='lines',
            name='Predicted',
            line=dict(dash='dash')
        ))

        fig.update_layout(
            title=dict(text='Actual vs Predicted', x=0.5),  
            xaxis=dict(title='Index', showgrid=True),
            yaxis=dict(title='Value', showgrid=True),
            showlegend=True,
            width=1200,  
            height=600   
        )

        pio.show(fig)

    def plot_metrics(self,metrics):
        """
        Plot error metrics comparison.

        Parameters:
            metrics (dict): A dictionary where the keys are the names of the metrics 
                            (e.g., 'MAE', 'MAPE', 'RMSE') and the values are the corresponding 
                            numerical values for those metrics.

        The function generates a bar chart comparing different error metrics (e.g., MAE, MAPE, RMSE),
        displaying each metric's name on the x-axis and its value on the y-axis.
        """
        go.Figure(go.Bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            text=[f'{v:.2f}' for v in metrics.values()],
            textposition='auto',
            marker_color='royalblue'
        )).update_layout(
            title="Error Metrics Comparison", 
            xaxis_title="Metric", 
            yaxis_title="Value", 
            template="plotly_dark", 
            bargap=0.65
        ).show()


# Example usage:
# vlstm = VLSTM(target=self.target)
# history, y_test, y_pred, train_score, test_score = vlstm.train(df, top_features)
# vlstm.summary()
# metrics = vlstm.evaluate(y_test, y_pred)
# vlstm.plot_metrics(metrics)
# predictions = vlstm.predict(new_df, features)
# vlstm.forecast(df,top_features,10,noise_factor=0.025) # Noise is added (if you want to simulate realistic variations in the forecast).
# vlstm.plot_forecast(df,top_features,30,noise_factor=0.025)
# vlstm.evaluate(y_test, y_pred)
# vlstm.plot_loss(history)
# vlstm.prediction_plot(y_test, y_pred)
