import pandas as pd
from pycaret.classification import predict_model, load_model
class PredictChurn:

    def load_data(filepath):
        """
        Loads churn data into a DataFrame from a string filepath.
        """
        df = pd.read_csv(filepath, index_col='customerID')
        return df

    def make_predictions(df):
        """
        Uses the pycaret best model to make predictions on data in the df dataframe.
        """
        model = load_model('gbc')
        predictions = predict_model(model, data=df)
        predictions.rename({'Label': 'of Churn', 'Score': 'Probability'}, axis=1, inplace=True)
        predictions['of Churn'].replace({1: 'Churn', 0: 'No Churn'},
                                                inplace=True)
        return predictions[{'of Churn', 'Probability'}]

    if __name__ == "__main__":
        df = load_data('new_churn_data.csv')
        predictions = make_predictions(df)
        print(predictions)