
# 🏏 Cricket Game Prediction System

## Project Overview
A machine learning application that predicts cricket outcomes using historical match commentary and cricket statistics.

The project includes two prediction systems:
1. Next-Ball Outcome Prediction
2. Match-Winner Prediction

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Natural Language Processing (NLP)
- TF-IDF and CountVectorizer
- Multinomial Naive Bayes
- Logistic Regression
- Streamlit
- Matplotlib and Seaborn

## Features

### 1. Next-Ball Prediction
Predicts the outcome of the next delivery using commentary from the previous three balls.

Possible outcomes include:
- No run
- Run
- Four
- Six
- Wicket
- Wide
- No ball
- Bye
- Leg bye

### 2. Match-Winner Prediction
Predicts Team 1 or Team 2 as the likely winner using:
- First-innings commentary
- Runs
- Wickets
- Balls
- Run rate
- First batting side
- Match format

## Machine Learning Algorithms

### Next-Ball Model
- Algorithm: Multinomial Naive Bayes
- Text features: Bag-of-Words with n-grams
- Test accuracy: 55.08%

### Match-Winner Model
- Algorithm: Logistic Regression
- Text features: TF-IDF
- Numeric features: StandardScaler
- Categorical features: OneHotEncoder
- Test accuracy: 53.36%
- Majority baseline: 57.46%

The match-winner model does not outperform the majority baseline on the current test dataset.

## Dataset
Cricket Scorecard and Commentary Dataset.

- Commentary files: 1,432
- Valid matches used for winner prediction: 1,338
- Winner model training records: 1,070
- Winner model testing records: 268

## How to Run

1. Install dependencies:

   pip install -r requirements.txt

2. Start the Streamlit application:

   streamlit run app.py

3. Open the application in your browser.

## Application Sections
- Home
- Next-Ball Prediction
- Match-Winner Prediction
- Model Performance

## Limitations
- Predictions are based on historical cricket data.
- Rare next-ball outcomes are difficult to predict.
- Match-winner predictions require first-innings information.
- The match-winner model has limited predictive accuracy.

## Future Improvements
- More balanced training data
- Additional match statistics
- Improved feature engineering
- More advanced machine learning algorithms
- Live cricket data integration

## Project Status
Completed and tested locally.
