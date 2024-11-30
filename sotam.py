import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Input
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import plotly.graph_objects as go
import plotly.io as pio

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

        allowed_losses = ['mae', 'mse', 'mape', 'mad', 'rmse']
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

        split_ratio = self.train_size
        split = int(split_ratio * len(X))

        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        gpus = tf.config.list_physical_devices('GPU')
        device = 'GPU' if gpus else 'CPU'
        print(f"Using device: {device}")

        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            self.model = self.build_model((self.sequence_length, X_train.shape[2]))

            callbacks = [ModelCheckpoint(self.best_model_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)]
            
            if self.early_stopping:
                callbacks.append(EarlyStopping(monitor='val_loss', patience=self.patience, verbose=1))

            self.history = self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.2, 
                                          verbose=1, callbacks=callbacks)
            
        self.model = load_model(self.best_model_path)

        y_pred = self.model.predict(X_test)
        y_train_pred = self.model.predict(X_train)

        y_test_actual = self.scalers[self.target].inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_actual = self.scalers[self.target].inverse_transform(y_pred).flatten()
        y_train_actual = self.scalers[self.target].inverse_transform(y_train.reshape(-1, 1)).flatten()
        y_train_pred_actual = self.scalers[self.target].inverse_transform(y_train_pred).flatten()

        train_score = r2_score(y_train_actual, y_train_pred_actual)*100
        test_score = r2_score(y_test_actual, y_pred_actual)*100

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
    
    def evaluate(self, y_actual, y_pred):
        """
        Evaluate the model's performance using MAE, RMSE, R², MAPE, and MAD metrics.

        Parameters:
            y_actual (np.ndarray): The actual target values.
            y_pred (np.ndarray): The predicted target values.

        Returns:
            dict: A dictionary containing MAE, RMSE, R², MAPE, and MAD scores.
        """
        # Ensure no division by zero in MAPE calculation
        non_zero_indices = y_actual != 0

        mae = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        r2 = r2_score(y_actual, y_pred)
        mape = (np.mean(np.abs((y_actual[non_zero_indices] - y_pred[non_zero_indices]) / y_actual[non_zero_indices])) * 100 
                if np.any(non_zero_indices) else np.inf)
        mad = np.mean(np.abs(y_actual - y_pred))

        print(f"Evaluation Metrics:\n"
              f"Mean Absolute Error (MAE): {mae:.4f}\n"
              f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%\n"
              f"Mean Absolute Deviation (MAD): {mad:.4f}\n"
              f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        
        return {'MAE': mae, 'MAPE': mape, 'MAD': mad, 'RMSE': rmse}
    
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

# Example usage:
# vlstm = VLSTM(target='Close')
# history, y_test, y_pred, train_score, test_score = vlstm.train(df, top_features)
# vlstm.summary()
# predictions = vlstm.predict(new_df, features)
# vlstm.evaluate(y_test, y_pred)
# vlstm.plot_loss(history)
# vlstm.prediction_plot(y_test, y_pred)
