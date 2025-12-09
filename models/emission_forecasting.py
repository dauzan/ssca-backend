import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context
import numpy as np
from datetime import datetime, timedelta

context.set_context(mode=context.PYNATIVE_MODE)
ms.set_device('CPU')

class EmissionForecastingModel(nn.Cell):
    """
    LSTM-Attention based emission forecasting model
    Simplified version of the proposal's multivariate model
    """
    def __init__(self, input_dim=11, hidden_dim=64, output_dim=1, num_layers=2):
        super(EmissionForecastingModel, self).__init__()
        
        # Input features: production_volume, energy_kwh, gas_m3, runtime_hours,
        #                 shipment_count, distance, modal_split, etc.
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.attention = nn.Dense(hidden_dim, 1)
        
        self.fc_out = nn.Dense(hidden_dim, output_dim)
        self.dropout = nn.Dropout(keep_prob=0.7)
        
    def construct(self, x):
        """
        x shape: (batch, sequence_length, input_dim)
        sequence_length: 30 days of historical data
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)
        
        # Simple attention (average pooling for simplicity in CPU mode)
        attn_weights = ops.Softmax(axis=1)(self.attention(lstm_out))
        context_vector = ops.ReduceSum()(lstm_out * attn_weights, 1)
        
        output = self.fc_out(self.dropout(context_vector))
        return output


class ProbabilisticEmissionModel(nn.Cell):
    """
    Wrapper to add uncertainty quantification
    """
    def __init__(self, base_model):
        super(ProbabilisticEmissionModel, self).__init__()
        self.base_model = base_model
        self.log_var_layer = nn.Dense(64, 1)
        
    def construct(self, x):
        mean = self.base_model(x)
        # Simplified: use a learnable log variance
        log_var = self.log_var_layer(x[:, -1, :])  # Last time step
        return mean, log_var


class EmissionForecaster:
    """
    High-level forecaster class for API usage
    """
    def __init__(self, model_path=None):
        self.model = EmissionForecastingModel()
        self.is_loaded = False
        
        if model_path:
            self.load_model(model_path)
        else:
            # Use demo mode with synthetic data
            self.demo_mode = True
            self.is_loaded = True
    
    def load_model(self, path):
        """Load pre-trained MindSpore checkpoint"""
        try:
            param_dict = ms.load_checkpoint(path)
            ms.load_param_into_net(self.model, param_dict)
            self.is_loaded = True
            print(f"Model loaded from {path}")
        except Exception as e:
            print(f" Could not load model: {e}. Using demo mode.")
            self.demo_mode = True
            self.is_loaded = True
    
    def predict(self, facility_id, horizon=30):
        """
        Generate emission forecast for next `horizon` days
        
        Returns:
            {
                'predictions': [day1, day2, ..., dayN],
                'confidence': [(lower, upper), ...],
                'dates': [date1, date2, ...]
            }
        """
        if self.demo_mode:
            return self._generate_demo_forecast(horizon)
        
        # In production: prepare input tensor from historical data
        # For now, generate synthetic input
        sequence_length = 30
        input_dim = 11
        
        # Create dummy input (replace with real historical data)
        x = Tensor(np.random.randn(1, sequence_length, input_dim), ms.float32)
        
        # Forward pass
        self.model.set_train(False)
        predictions = []
        
        for i in range(horizon):
            pred = self.model(x)
            predictions.append(float(pred.asnumpy()[0, 0]))
            
            # Sliding window: shift input (simplified)
            # In production: properly update with actual new observations
        
        # Generate confidence intervals (simplified)
        confidence = [(p * 0.9, p * 1.1) for p in predictions]
        
        # Generate dates
        dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') 
                 for i in range(1, horizon + 1)]
        
        return {
            'predictions': predictions,
            'confidence': confidence,
            'dates': dates
        }
    
    def _generate_demo_forecast(self, horizon):
        """Generate synthetic forecast for demo purposes"""
        base = 62500
        predictions = []
        
        for i in range(horizon):
            # Add trend + seasonality + noise
            value = base * (1 + np.sin(i / 5) * 0.08 + i / 200 + np.random.randn() * 0.02)
            predictions.append(max(0, value))
        
        confidence = [(p * 0.93, p * 1.07) for p in predictions]
        dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') 
                 for i in range(1, horizon + 1)]
        
        return {
            'predictions': predictions,
            'confidence': confidence,
            'dates': dates
        }
    
    def train(self, train_data, epochs=50, lr=0.001):
        """
        Train the model on historical data
        
        train_data: numpy array of shape (num_samples, sequence_length, input_dim)
        """
        # Define loss and optimizer
        loss_fn = nn.MSELoss()
        optimizer = nn.Adam(self.model.trainable_params(), learning_rate=lr)
        
        # Training loop
        def forward_fn(data, label):
            pred = self.model(data)
            loss = loss_fn(pred, label)
            return loss
        
        grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)
        
        @ms.jit
        def train_step(data, label):
            loss, grads = grad_fn(data, label)
            optimizer(grads)
            return loss
        
        self.model.set_train(True)
        
        for epoch in range(epochs):
            x = Tensor(train_data['X'], ms.float32)
            y = Tensor(train_data['y'], ms.float32)
            
            loss = train_step(x, y)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.asnumpy():.4f}")
        
        print("Training complete")
        self.is_loaded = True