# IMDB Video Game Ratings Analysis

A machine learning analysis to understand the relationship between video game genres, certificates, and their IMDB ratings. This project implements multiple ML approaches including KNN, Decision Trees, Logistic Regression, and SVC to predict game ratings based on genre and certificate data.

## Project Overview

This project analyzes a dataset of video game ratings from IMDB to investigate whether genre and certificate ratings have a significant impact on overall game ratings. Through various machine learning models and PCA, we explore these relationships and their predictive capabilities.

### Project Presentation Materials

#### Video Presentation
[![IMDB Video Game Ratings](https://img.youtube.com/vi/f8cvHWCq0Ag/maxresdefault.jpg)](https://www.youtube.com/watch?v=f8cvHWCq0Ag)

#### Slide Deck
📄 View our detailed presentation slides or [download the PDF directly](Project%20Presentation.pdf).

## Dataset

The dataset is sourced from Kaggle: [IMDB Video Games Dataset](https://www.kaggle.com/datasets/muhammadadiltalay/imdb-video-games)
- ~20,000 video game entries
- Features include: name, year, certificate, rating, votes, and genre information
- Preprocessed to remove entries with less than 100 votes
- Genre data is represented as boolean values
- Certificate ratings are one-hot encoded

## Repository Structure
```
├── data/
│   └── imdb-videogames.csv
├── notebooks/
│   └── imbd_videogames.ipynb
├── src/
│   └── model_training.py
├── Project Presentation.pdf
├── README.md
└── requirements.txt
```

## Setup & Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/video-game-ratings-analysis.git
cd video-game-ratings-analysis
```

2. Download the dataset:
   - Visit [IMDB Video Games Dataset](https://www.kaggle.com/datasets/muhammadadiltalay/imdb-video-games)
   - Download `imdb-videogames.csv`
   - Place it in the `data/` directory

3. Set up your environment:
```bash
# Using pip
pip install -r requirements.txt

# Using conda
conda create --name vgames python=3.8
conda activate vgames
pip install -r requirements.txt
```

## Running the Analysis

### Using Python
```bash
python src/model_training.py
```

### Using Jupyter Notebook
```bash
jupyter notebook notebooks/imbd_videogames.ipynb
```

You can also view the analysis in Google Colab

## Key Findings

- The Logistic Regression model achieved the highest accuracy at 45.72%
- Genre and certificate data alone are not strong predictors of game ratings
- Action and Adventure are the most common genres in the dataset
- Data shows weak correlations between genres/certificates and ratings
- Model performance significantly exceeds random guessing (10% baseline)

## Models Implemented
- K-Nearest Neighbors (with n=2, 5, 8)
- Decision Trees (with max_depth=3, 5, 8)
- Logistic Regression
- Support Vector Classification (SVC)

## Data Processing
- Removal of duplicates and null values
- One-hot encoding of certificate data
- Filtering games with <100 votes
- Principal Component Analysis (PCA) for dimension reduction
- Stratified K-Fold cross-validation

## Contributors
- Jacob Russell
- Croix Westbrock
- Blake Nelson

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- IMDB for the video game ratings data
- Muhammad Adil Talay for creating and maintaining the dataset on Kaggle
