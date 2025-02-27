# Match Score Prediction

This project utilizes machine learning models to predict football match scores. Separate score prediction models have been developed for major European leagues (England, France, Germany, Italy, Netherlands, Spain) and the Champions League. Users can view the predicted score for a match by entering the home and away teams.

## Project Features

-   **Football Match Score Prediction:** Users can view the predicted match score for home and away teams in a selected league.
-   **Machine Learning Models:** Separate RandomForestRegressor models have been trained for each league. These models make score predictions based on features derived from past match results and team statistics.
-   **User-Friendly Interface:** An interactive user interface built with Streamlit allows users to easily make predictions.

## Requirements

To run the project, you need to install the following libraries:

-   pandas
-   numpy
-   joblib
-   sklearn
-   streamlit

To install these libraries:

```bash
pip install pandas numpy joblib scikit-learn streamlit
```

## Usage
Step 1: Download and Run the Project
You can clone or download the project to your computer:
```bash
git clone https://github.com/omerf/Match-Result-Prediction.git
cd Match-Result-Prediction
```

Step 2: Set Up Datasets and Models
You need to place the model files and datasets in the following directories:

Models: Model files named league_home.pkl and league_away.pkl should be in the model/models/ directory for each league.
Datasets: Appropriate CSV files for each league (e.g., bundesliga_train.csv, cl_train.csv, etc.) should be in the train_data/data/ directory.

Step 3: Run the Application
To start the Streamlit application, run the following command:

```bash
streamlit run user_interface.py
```
This command will start the Streamlit interface on a local server and open it in your browser.

Step 4: Make Match Prediction
Once the application is open, users can select a league and enter the home and away teams to view the predicted score.

Model Training Steps
If you want to retrain the models, you can follow these steps:

- Data Preparation: Preprocess the data using appropriate data for each league. Process the "Home Team", "Away Team", "Home Score", and "Away Score" columns appropriately for the model.
- Model Training: Train separate models for home and away teams for each league using a model like RandomForestRegressor.
-Model Saving: Save the models using joblib after training. For example:

Python
```bash
import joblib
```
# Saving the model

``` bash
joblib.dump(home_model, 'path_to_save_home_model.pkl')
joblib.dump(away_model, 'path_to_save_away_model.pkl')
```

# Used Models
In this project, the RandomForestRegressor model is used to predict match scores. The model makes score predictions based on past match results and team statistics.

# Datasets
The datasets include past match results and team statistics for each league. These data are used to make predictions by analyzing the relationships between match results and home/away teams.

# Development
If you want to contribute to the project, you can follow these steps:
- Fork the repository.
- Add new features or fix bugs.
- Submit your changes as a pull request.

# License
This project is licensed under the MIT License.
