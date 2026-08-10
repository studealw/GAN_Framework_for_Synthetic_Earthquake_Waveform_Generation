import pandas as pd
from scipy.signal import butter, filtfilt

df = pd.read_csv('test_data.csv')

df["Acceleration"] = df["Acceleration"] - df["Acceleration"].mean()

std = df["Acceleration"].std().item()
df["Acceleration"]= df["Acceleration"]/std

def apply_butterworth(data, cutoff_freq, sample_rate, order=4):
    # Calculate the Nyquist frequency (half the sample rate)
    nyquist = 0.5 * sample_rate
    # Normalize the cutoff frequency for the algorithm
    normal_cutoff = cutoff_freq / nyquist
    
    # Generate the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # Apply the filter using filtfilt (this prevents the wave from shifting left/right)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

df["Filtered_Acceleration"] = apply_butterworth(df["Acceleration"], cutoff_freq=20, sample_rate=100)

Absolute_Filtered_Acceleration= df["Filtered_Acceleration"].abs()
STA = Absolute_Filtered_Acceleration.rolling(window=100 , min_periods=1).mean()
LTA = Absolute_Filtered_Acceleration.rolling(window=400, min_periods=1).mean()
df["STA_LTA"] = STA / LTA