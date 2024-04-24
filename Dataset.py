import pandas as pd 
import numpy as np

np.random.seed(42)  # For reproducibility

# Generate data
num_patients = 100
patient_ids = np.random.randint(10000, 99999, size=num_patients)
sex = np.random.choice([0, 1, 2], size=num_patients)
race = np.random.choice([0, 1, 2, 3, 4, 5], size=num_patients)
socioeconomic_status = np.random.uniform(0, 1, size=num_patients)

# Create a DataFrame
patient_data = pd.DataFrame({
    "Patient ID": patient_ids,
    "Sex": sex,
    "Race": race,
    "Socioeconomic Status": socioeconomic_status
})

# Specify your desired file path
file_path = 'C:/TrialUnity/patient_dataset.csv'

# Save the DataFrame to a CSV file
patient_data.to_csv(file_path, index=False)
