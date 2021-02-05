from src.game import beats, loses_to
from src.database import query_to_df
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from random import randint
from joblib import dump, load
import tflite_runtime.interpreter as tflite

# implement model scoring
def score_model(model, record, drop_first=1):        
    """ Returns a score for a model's performance based on its record """

    n = len(record) - drop_first
    
    if n < 1:
        return 1.0

    # creates unweighted score
    wins = 0
    losses = 0
    for j in range(drop_first,len(record)):
        if record.iloc[j, 4+model] == beats(record.loc[j, 'p1']):
            wins = wins + 1
        elif record.iloc[j, 4+model] == loses_to(record.loc[j, 'p1']):
            losses = losses + 1
    #print(f"Wins: {wins}, Losses: {losses}, Ties: {ties}")
    model_score = (wins - losses) / (n)

    # creates a weighted model score that prioritizes recency
    model_record = []
    w_max = 0
    for j in range(drop_first,len(record)):
        if n == 1 and j == 0:
            return model_score
        if record.iloc[j, 4+model] == beats(record['p1'].iloc[j]):
            model_record.append(j**2)
        elif record.iloc[j, 4+model] == loses_to(record['p1'].iloc[j]):
            model_record.append(-j**2)
        w_max = w_max + j**2
    weighted_model_score = np.sum(model_record) / w_max
    
    return weighted_model_score

def vectorize(choice1, choice2):
    """ returns a vector describing the difference between one choice and another """
    if choice1 == beats(choice2):
        return 1
    elif choice1 == loses_to(choice2):
        return -1
    else:
        return 0

def build_xrecord(record):
    """builds xrecord that includes choice vectors"""
    xrecord = record.copy()
    xrecord = xrecord.drop(['game_id', 'timestamp', 'ip_address'], axis=1, errors='ignore')
    
    if 'round' not in xrecord.columns:
        xrecord = xrecord.reset_index(drop=False)
        xrecord = xrecord.rename(columns={'index':'round'})

    xrecord['p1_vector_from_prev'] = None
    xrecord['p2_vector_from_prev'] = None
    xrecord['p1_vector_from_prev_p2'] = None
    xrecord['p2_vector_from_prev_p1'] = None
    xrecord['p1_next_choice'] = None

    for i in range(len(xrecord)-1):
        xrecord['p1_vector_from_prev'].iloc[i+1] = vectorize(xrecord['p1'].iloc[i+1],xrecord['p1'].iloc[i])
        xrecord['p2_vector_from_prev'].iloc[i+1] = vectorize(xrecord['p2'].iloc[i+1],xrecord['p2'].iloc[i])
        xrecord['p1_vector_from_prev_p2'].iloc[i+1] = vectorize(xrecord['p1'].iloc[i+1],xrecord['p2'].iloc[i])
        xrecord['p2_vector_from_prev_p1'].iloc[i+1] = vectorize(xrecord['p2'].iloc[i+1],xrecord['p1'].iloc[i])
        xrecord['p1_next_choice'].iloc[i] = xrecord['p1'].iloc[i+1]
    return xrecord.astype(float, errors='ignore')

def build_xrecord_onehot(record, length=5):
  xrecord_onehot = build_xrecord(record)

  enc = OneHotEncoder(categories=[range(3),range(3),range(3),range(3),range(3),range(3),range(3),range(3),range(3),range(6)])
  df = pd.DataFrame(enc.fit_transform(xrecord_onehot[['p1', 'p2', 'winner',	'model0',	'model1',	'model2',	'model3',	'model4',	'model5', 'model_choice']]).toarray(), columns=[name[:-2] for name in enc.get_feature_names(['p1', 'p2', 'winner',	'model0',	'model1',	'model2',	'model3',	'model4',	'model5', 'model_choice'])])
  xrecord_onehot = xrecord_onehot.merge(df, left_index=True, right_index=True).drop(['p1', 'p2', 'winner','model0',	'model1',	'model2',	'model3',	'model4',	'model5', 'model_choice'], axis=1)
  
  xrecord_onehot['valid_round'] = xrecord_onehot['round'] >= length
  xrecord_onehot['valid_round'] = xrecord_onehot['valid_round'].astype(float)
  
  return xrecord_onehot

def get_Xy(record):
    """Builds and returns X, y, and the this_round for use in machine learning models"""
    xrecord = build_xrecord(record)

    X = pd.DataFrame(columns = ['round', 'p1', 'p2', 'winner', 'p1_vector_from_prev', 
                                'p2_vector_from_prev', 'p1_vector_from_prev_p2',
                                'p2_vector_from_prev_p1', 'p1_prev_vector_from_prev', 
                                'p2_prev_vector_from_prev'])
    y = pd.Series(dtype=int)

    for i in range(1, len(record)):
        x={}
        try: x['round'] = xrecord.iloc[i].name[-1]
        except: x['round'] = xrecord.iloc[i].name
        x['p1'] = xrecord['p1'].iloc[i]
        x['p2'] = xrecord['p2'].iloc[i]
        x['winner'] = xrecord['winner'].iloc[i]
        x['p1_vector_from_prev'] = xrecord['p1_vector_from_prev'].iloc[i]
        x['p2_vector_from_prev'] = xrecord['p2_vector_from_prev'].iloc[i]
        x['p1_vector_from_prev_p2'] = xrecord['p1_vector_from_prev_p2'].iloc[i]
        x['p2_vector_from_prev_p1'] = xrecord['p2_vector_from_prev_p1'].iloc[i]
        x['p1_prev_vector_from_prev'] = xrecord['p1_vector_from_prev'].iloc[i-1]
        x['p2_prev_vector_from_prev'] = xrecord['p2_vector_from_prev'].iloc[i-1]

        try: next_is_new_game = xrecord.iloc[i+1].name[-1] == 0
        except: next_is_new_game = False
        if x['round'] not in [0,1] and i != len(xrecord)-1 and next_is_new_game == False:
            X = X.append(x, ignore_index=True)
            y = y.append(pd.Series({'index':xrecord['p1_next_choice'].iloc[i]}), ignore_index=True)
    
    this_round = X.iloc[-1]
    X = X[:-1]
    y = y[:-1].astype('int')

    return X,y,this_round

def get_nn_Xy(record, length=7):
    """
    Builds an three dimensional array X that will input a 2 dimensional array for each value of y, which represents the player's next choice
    """

    xrecord = build_xrecord_onehot(record, length=length)
    print('xrecord')
    print(xrecord)

    # builds X and y based off of summary statistics of the previous (length) rounds
    X = np.ndarray((len(record)-length,length, xrecord.shape[1]-1))
    y = pd.Series(dtype=int)
    this_round = np.ndarray((1,length, xrecord.shape[1]-1))

    # append a (length x rows) matrix to each X index, and its corresponding next round value to y
    for i in range(length, len(record)):
      X[i-length] = xrecord[i-length+1:i+1].drop('p1_next_choice',axis=1).to_numpy()
      y = y.append(pd.Series({'index':xrecord['p1_next_choice'].iloc[i]}), ignore_index=True)

    this_round[0] = X[-1]
    X = X[:-1]
    y = y[:-1].astype('int')

    return X,y,this_round

# define models
def model0(record):
    """ Chooses the choice that would lose to or beat the player's last choice. Based on whether a player is changing their answers or not in the past three rounds. """
    if len(record) > 3:
        repeats = 0
        for i in range(1,3):
            repeats = repeats + (int(record['p1'].iloc[-i] == record['p1'].iloc[-i-1])) 
        if repeats > 1:
            choice = beats(record['p1'].iloc[-1])
        else:
            choice = loses_to(record['p1'].iloc[-1])
    elif len(record) > 0:
        choice = loses_to(record['p1'].iloc[-1])
    else: choice = int(randint(0,2))
    return int(choice)


def model1(record):
    """ Vector-based choice based on past three rounds """
    if len(record) > 1:
        if len(record) > 5:
            vectors = []
            for i in range(1,5):
                vectors.append(vectorize(record['p1'].iloc[-i], record['p1'].iloc[-i-1]))
            vector = pd.Series(vectors).value_counts().index[0]
        else:
            vector = vectorize(record['p1'].iloc[-1], record['p1'].iloc[-2])
        if vector == 1:
            choice = beats(beats(record['p1'].iloc[-1]))
        elif vector == -1:
            choice = beats(loses_to(record['p1'].iloc[-1]))
        elif vector == 0:
            choice = beats(record['p1'].iloc[-1])
    else:
        choice = 0
    return int(choice)


def model2(record):
    """ Chooses the choice that would beat the player's most frequent recent choice. Based on repeated choices """
    if len(record) >= 5:
        most_freq = record['p1'].value_counts().iloc[-5:].index[0]
        choice = beats(most_freq)
    elif len(record) > 0:
        most_freq = record['p1'].value_counts().index[0]
        choice = beats(most_freq)
    else: choice = 0
    return int(choice)


def model3(record):
    """ Chooses the choice that would beat the player's least frequent recent choice"""
    if len(record['p1']) > 4:
        least_freq = record['p1'].value_counts().iloc[-5:].index[-1]
        choice = beats(least_freq)
    elif len(record['p1']) > 0:
        least_freq = record['p1'].value_counts().index[-1]
        choice = beats(least_freq)
    else: choice = 0
    return int(choice)


def model4(record):
    """ Uses a pickled tensorflow neural network model trained with historical data to predict next choice"""
    if len(record) < 8:
        return int(0)

    X,y,this_round = get_nn_Xy(record)

    nn_clf = load('nn_clf.joblib')
    curr = sk_flatten(this_round)

    guess = int(nn_clf.predict(curr))

    return beats(guess)


def model5(record):
    """ uses an sklearn decision tree model trained with historical data to predict next choice"""
    if len(record) < 5:
        return int(0)

    model = load(filename='decision_tree_clf.joblib')

    X,y,this_round = get_Xy(record)
    guess = int(model.predict(this_round.to_numpy().reshape(1, -1)))

    return beats(guess)


def build_historical_dtclf():    
    """ Builds the decision tree classifier for model 5 from historical data """
    query = """
    SELECT 1 AS record, game_id, n, p1, p2, winner, model_choice, model0, model1, model2, model3, model4
    FROM record_test 
    UNION 
    SELECT 2 AS record, game_id, n, p1, p2, winner, model_choice, model0, model1, model2, model3, model4
    FROM record_test2 
    UNION 
    SELECT 3 AS record, game_id, round, p1, p2, winner, model_choice, model0, model1, model2, model3, model4
    FROM record_test3
    UNION 
    SELECT 4 AS record, game_id, round, p1, p2, winner, model_choice, model0, model1, model2, model3, model4
    FROM record_test4
    UNION 
    SELECT 5 AS record, game_id, round, p1, p2, winner, model_choice, model0, model1, model2, model3, model4
    FROM record
    """

    #record = query_to_df(query).set_index(['record', 'game_id', 'n']).sort_index()
    record = pd.read_csv('rps-record_dtclf.csv', index_col='Unnamed: 0')

    X,y,this_round = get_Xy(record)

    dtclf = DecisionTreeClassifier(max_depth=500, max_features='auto', min_samples_split=5, splitter='best')
    dtclf.fit(X,y)
    dump(dtclf, 'decision_tree_clf.joblib')

    print('dtclf file updated')
    
    return

def sk_flatten(X):
        return X.reshape(X.shape[0], (X.shape[1]*X.shape[2]))

def build_nn():    
    """ Builds the neural network classifier for model 4 from historical data """
    query = """
    SELECT *
    FROM record
    """

    record = query_to_df(query)
    print('Record')
    print(record)
    #record = pd.read_csv('rps-record122.csv', index_col='Unnamed: 0')
    
    X,y,this_round = get_nn_Xy(record, length=7)

    X_flat = sk_flatten(X)

    nn_clf = MLPClassifier(hidden_layer_sizes=(50,10),
                        activation='relu',
                        solver='adam', 
                        learning_rate='adaptive', learning_rate_init=0.003, 
                        max_iter=200, 
                        alpha=0.1)

    nn_clf.fit(X_flat, y)

    dump(nn_clf, 'nn_clf.joblib')

    print('nn file updated')
    
    return

#currently not in use
def always_changing(record): return len(record) > 5 and False not in [record['p1'].iloc[-i] != record['p1'].iloc[-i-1] for i in range(1,6)]

def highest_score_model(model_scores): return int(model_scores.index(max(model_scores)))



def computer_choice(record):
    """ Makes a choice given the record of previous rounds """

    #need to move these into execution file so that I can save them 
    model_choices = []
    model_scores = []
    
    model_choices.append(model0(record))
    model_choices.append(model1(record))
    model_choices.append(model2(record))
    model_choices.append(model3(record))
    model_choices.append(model4(record))
    model_choices.append(model5(record))

    model_scores.append(score_model(0,record, drop_first=0))
    if len(record) >= 2:
        model_scores.append(score_model(1,record,drop_first=1))
        model_scores.append(score_model(2,record,drop_first=1))
        model_scores.append(score_model(3,record,drop_first=1))
    if len(record) >= 5:
        model_scores.append(score_model(4,record, drop_first=7)+0.15)
        model_scores.append(score_model(5,record, drop_first=5)+0.15)
        
    
    if len(record) < 1:
        model = 0
    elif record['winner'].iloc[-1] == 2:
        model = int(record['model_choice'].iloc[-1])
    else:
        model = highest_score_model(model_scores)
    
    print(f'Model Choices: {model_choices}')
    print(f'Model Scores: {model_scores}')
    print(f"Model {model} chosen.")
    
    # next: build a ensembler that aggregates model suggestions weighted by their scores to choose
    
    choice = model_choices[model]
    
    return choice, model, model_choices
