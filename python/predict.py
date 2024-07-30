import numpy as np
import pandas as pd
import optparse

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def predict(data: pd.DataFrame) -> pd.DataFrame:
    """
    Make predictions on a new dataset.
    
    Args:
    -----
    data: pandas DataFrame
        Data frame with input data. Contains the customernumber as row
        identifyer and the raw features.    
    
    Returns:
    --------
    predictions: pd.DataFrame
        Data frame with a prediction for all customers from the input data.
        Must contain only the columns ['customernumber', 'prediction'],
        where the prediction is either 0 (non-rebuyer, i.e. sent voucher) 
        or 1 (rebuyer, i.e. don't sent voucher).
    
    """

    # Store customernumbers
    customernumbers = data.customernumber

   ### YOUR CODE START ###
    data = data.drop(['customernumber', 'date', 'datecreated', 'deliverydatepromised', 'deliverydatereal'], axis=1)
    data = data.drop([ 'delivpostcode', 'advertisingdatacode', 'invoicepostcode'], axis=1)
    data = pd.get_dummies(data, columns=['salutation', 'domain', 'paymenttype', 'case'])
    cols = data.columns

    train = pd.read_csv("data/train.csv", sep=';', low_memory=False)
    train = train.drop(['customernumber', 'date', 'datecreated', 'deliverydatepromised', 'deliverydatereal'], axis=1)
    train = train.drop([ 'delivpostcode', 'advertisingdatacode', 'invoicepostcode'], axis=1)
    train = pd.get_dummies(train, columns=['salutation', 'domain', 'paymenttype', 'case'])
    cols2 = train.columns
    
    df_input, df_train = data.align(train, join='right', axis=1, fill_value=0)

    X = train.drop('target90', axis=1)
    y = train['target90']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=71)
    
    

    df_input = df_input.drop('target90', axis=1)

    model_best= RandomForestClassifier(class_weight='balanced', max_depth=9, min_samples_split=12, min_samples_leaf=17, max_features=5, n_estimators=37, random_state=71)
    model_best.fit(X_train, y_train)
    predictionss = model_best.predict(df_input)
    # Some dummy predictions (reference model: send every customer a voucher)
    predictions = pd.DataFrame({
        'customernumber': customernumbers,
        'prediction': predictionss  # Replace with your predictions
    })
    ### YOUR CODE END ###

    return predictions


if __name__ == '__main__':

    # Command line arguments
    optParser = optparse.OptionParser()
    optParser.add_option('-i', '--inputfile', action='store', type='string',
                         dest='input_file',
                         help='data/test_example.csv')
    optParser.add_option('-o', '--outputfile', action='store', type='string',
                         dest='output_file',
                         help='data/prediction.csv')
    opts, args = optParser.parse_args()

    # Read input data
    input_data = pd.read_csv(opts.input_file, sep=';', low_memory=False)

    # Apply prediction function
    predictions = predict(data=input_data)

    # Check if predictions are valid
    if type(predictions) != pd.DataFrame:
        raise ValueError('Predictions must be a data frame. '
                         f'Returned object is of type {type(predictions)}.')
    if set(predictions.columns) != {'customernumber', 'prediction'}:
        raise ValueError('Prediction function must return a data frame with '
                         'columns [\'customernumber\', \'prediction\']')
    if set(predictions.customernumber) != set(input_data.customernumber):
        raise ValueError('Customers must be the same as in input file.')
    if not predictions.prediction.isin([0, 1]).all():
        raise ValueError('All predictions must be either 0 or 1.')

    # Save predictions to output file
    predictions.to_csv(opts.output_file, sep=';', index=False)
