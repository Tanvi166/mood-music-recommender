# Mood-Based Music Recommender System

A modern Streamlit college project that recommends songs based on mood, energy level, and genre using unsupervised machine learning with K-Means clustering.

## Features

- Python + Streamlit single-file app
- Modern dark UI with gradient header
- Sidebar inputs for mood, energy level, and genre
- In-code dataset for the Streamlit app
- CSV dataset for the training notebook
- LabelEncoder preprocessing
- K-Means clustering from scikit-learn
- Top 5 song recommendations
- Recommendation confidence percentage
- Mood Meter
- Currently Recommended Vibe section
- Plotly scatter chart showing clusters
- Training notebook for ML workflow explanation
- No external APIs or Spotify authentication required

## Tech Stack

- Python
- Streamlit
- pandas
- NumPy
- scikit-learn
- Plotly
- Matplotlib
- Seaborn

## Project Structure

```text
mood-music-recommender/
|-- app.py
|-- songs.csv
|-- training.ipynb
|-- requirements.txt
|-- README.md
```

## How to Run Locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the Streamlit app:

```bash
python -m streamlit run app.py
```

3. Open the local URL shown in the terminal.

## Deployment on Streamlit Cloud

1. Push this project to a GitHub repository.
2. Go to Streamlit Cloud.
3. Create a new app from the repository.
4. Set the main file path to:

```text
app.py
```

5. Deploy the app.

## Machine Learning Approach

The app uses song data with categorical features:

- Mood
- Energy level
- Genre

These values are encoded using `LabelEncoder`, scaled using `StandardScaler`, and grouped using `KMeans`. When a user selects a mood, energy level, and genre, the app predicts the nearest cluster and ranks songs from that cluster by similarity.

## Training Notebook

The `training.ipynb` notebook explains the complete machine learning workflow:

- Loading `songs.csv`
- Displaying the first 5 rows
- Checking dataset information and null values
- Encoding mood, genre, and energy with `LabelEncoder`
- Selecting clustering features
- Training K-Means with 4 clusters
- Creating an elbow method graph
- Visualizing song clusters with a scatter plot
- Explaining the final unsupervised learning conclusion

## Footer

Developed using Streamlit & Machine Learning
