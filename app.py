import streamlit as st
import torch
import torch.nn as nn
import librosa
import numpy as np
import torch.nn.functional as F


st.markdown("<h2 style='text-align: center; color: blue;'>Music Genre Prediction App</h2>", unsafe_allow_html=True)
st.markdown("Upload an audio file to predict its genre. Supported genres include: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock.")



# Define GenreClassifier
class GenreClassifierCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(GenreClassifierCNN, self).__init__()
        
        # First Convolutional Block
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Second Convolutional Block
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Third Convolutional Block
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Fully Connected Layers with Dropout
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 19, 512),
            nn.ReLU(),
            nn.Dropout(0.5)  # 50% dropout
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5)  # 50% dropout
        )
        self.fc3 = nn.Linear(256, num_classes)  
        
    def forward(self, x):
        # Add channel dimension if missing
        if x.dim() == 2:  
            x = x.unsqueeze(1)  

        # Convolutional and Pooling Layers
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)  
        return x


# Load model
model = GenreClassifierCNN()  
model.load_state_dict(torch.load("genre.pth"))
model.eval()

# Function to extract features
labels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

def extract_features(audio_path, target_length=153):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs = np.mean(mfccs.T, axis=0)
    
    # Pad or truncate to target_length
    if len(mfccs) < target_length:
        mfccs = np.pad(mfccs, (0, target_length - len(mfccs)), 'constant')
    else:
        mfccs = mfccs[:target_length]
    return mfccs

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Process uploaded audio directly
    features = extract_features(uploaded_file)  
    input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0) 

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        prediction = output.argmax().item()

    st.write(f"Predicted Genre: {labels[prediction]}")
