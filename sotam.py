import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Input
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import plotly.graph_objects as go
import plotly.io as pio

class VLSTM:
    def __init__(self, target, sequence_length=30, batch_size=48, n1=300, n2=200, d1=25, d2=1, epochs=10, learning_rate=0.001, 
                 Dropout = 0.0, train_size=0.8, best_model_path='best_model.keras', loss='mae', activation='linear'):
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.n1 = n1
        self.n2 = n2
        self.d1 = d1
        self.d2 = d2
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.Dropout = Dropout
        self.train_size = train_size
        self.target = target
        self.model = None
        self.best_model_path = best_model_path
        self.scalers = {}
        self.activation = activation
        self.history = None

        allowed_losses = ['mae', 'mse', 'mape','mad']
        if loss in allowed_losses:
            self.loss = loss
        else:
            raise ValueError(f"Invalid loss function. Allowed values are {allowed_losses}")

    def create_sequences_optimized(self, data):
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
        data = df[features].copy()
        for feature in features:
            scaler = MinMaxScaler(feature_range=(0, 1))
            data[feature] = scaler.fit_transform(data[[feature]])
            self.scalers[feature] = scaler

        self.target_idx = features.index(self.target)
        return self.create_sequences_optimized(data)

    def build_model(self, input_shape):
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(LSTM(self.n1, return_sequences=True))
        model.add(LSTM(self.n2, return_sequences=False))
        model.add(Dense(self.d1))
        model.add(Dropout(self.Dropout))
        model.add(Dense(self.d2, activation=self.activation))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=self.loss)
        return model

    def train_and_evaluate(self, df, features):
        X, y = self.prepare_data(df, features)

        split_ratio = self.train_size
        split = int(split_ratio * len(X))

        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        self.model = self.build_model((self.sequence_length, X_train.shape[2]))

        model_checkpoint = ModelCheckpoint(self.best_model_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)

        self.history = self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.2, 
                                    verbose=1, callbacks=[model_checkpoint])

        self.model = load_model(self.best_model_path)

        y_pred = self.model.predict(X_test)
        y_train_pred = self.model.predict(X_train)

        y_test_actual = self.scalers[self.target].inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_actual = self.scalers[self.target].inverse_transform(y_pred).flatten()
        y_train_actual = self.scalers[self.target].inverse_transform(y_train.reshape(-1, 1)).flatten()
        y_train_pred_actual = self.scalers[self.target].inverse_transform(y_train_pred).flatten()

        r2_train = r2_score(y_train_actual, y_train_pred_actual)
        r2_test = r2_score(y_test_actual, y_pred_actual)

        return self.history, y_test_actual, y_pred_actual, r2_train, r2_test

    def predict(self, data, features):
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

    def plot_loss(self, history):
        if not self.history:
            raise ValueError("No training history found. Train the model before plotting.")
        
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=list(range(len(history.history['loss']))), 
            y= history.history['loss'], 
            mode='lines+markers', 
            name='Loss History', 
            line=dict(color='cadetblue')))
        
        fig.add_trace(go.Scatter(
            x=list(range(len(history.history['val_loss']))),
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
            xaxis=dict(tickmode='array', tickvals=list(range(len(history.history['loss']) + 1))),
            yaxis=dict(gridcolor='lightgrey'),
            plot_bgcolor='white')
        fig.show()

###

    def prediction_plot(self, y_test, y_pred):
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
# lstm = LSTM_Model(target='Close')
# history, y_test_actual, y_pred_actual, r2_train, r2_test = lstm.train_and_evaluate(data, features)
# predictions = lstm.predict(new_data, features)
# lstm.plot()