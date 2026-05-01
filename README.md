# Sentiment Analysis of IMDB Movie Reviews

This repo compares two approaches for binary sentiment classification on IMDB movie reviews:

- `tyler_lstm_final.ipynb`: cleaned LSTM workflow with final accuracy, epoch-by-epoch losses, hyperparameters, and graph outputs.
- `david_distilbert_final.ipynb`: DistilBERT fine-tuning workflow with training curves, final metrics, confusion matrix, and ROC curve.
- `Sentiment_Analysis_of_IMDB_Movie_Reviews_Using_LSTM_and_DistilBERT.ipynb`: original combined LSTM/EDA notebook kept for reference.

## Recommended Workflow

1. Run `tyler_lstm_final.ipynb` for the LSTM baseline.
2. Run `david_distilbert_final.ipynb` for the transformer model.
3. Use the saved CSV files in `outputs/` and saved plots in `figures/` for the final report or presentation.

## Tyler's Reporting Checklist

The cleaned LSTM notebook now has dedicated sections for:

- Final test accuracy: `outputs/lstm_test_metrics.csv`
- Training and validation loss for each epoch: `outputs/lstm_training_history.csv`
- Final hyperparameters: the `## Hyperparameters` section in `tyler_lstm_final.ipynb`

## Expected Graphs

The cleaned LSTM notebook saves:

- `figures/lstm_class_distribution.png`
- `figures/lstm_review_length_distribution.png`
- `figures/lstm_training_curves.png`
- `figures/lstm_confusion_matrix.png`
- `figures/lstm_roc_curve.png`

The DistilBERT notebook already saves comparable model-performance figures.

## Dataset

The repo expects `IMDB Dataset.csv` at the project root. The CSV is the Kaggle IMDB movie review dataset with `review` and `sentiment` columns.
