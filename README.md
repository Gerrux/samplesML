# samplesML

from joblib import Parallel, delayed
import joblib
  
  
# Save the model as a pickle in a file
joblib.dump(knn, 'filename.pkl')
  
# Load the model from the file
knn_from_joblib = joblib.load('filename.pkl')
  
# Use the loaded model to make predictions
knn_from_joblib.predict(X_test)
