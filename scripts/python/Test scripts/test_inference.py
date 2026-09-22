import numpy as np

from inference.predictor import predict_risk


sample = np.random.randn(12, 69).astype(np.float32)


risk = predict_risk(sample)


print("Predicted sepsis risk:", risk)
