# build AI decision function that returns the decision and the model used to make it, then build a 
# record dataframe that includes (p1, p2, winner, model)

from src.game import beats, loses_to
from src.database import query_to_df
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from random import randint
from joblib import dump, load
import tensorflow as tf
import keras

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

def build_xrecord_onehot(record):
  xrecord_onehot = build_xrecord(record)
  xrecord_onehot = xrecord_onehot.drop('round', axis=1, errors='ignore')

  xrecord_onehot = xrecord_onehot.merge(pd.DataFrame(tf.one_hot(xrecord_onehot['p1'],3).numpy(), columns=['p1_0', 'p1_1', 'p1_2']), left_index=True, right_index=True).drop(['p1'], axis=1)
  xrecord_onehot = xrecord_onehot.merge(pd.DataFrame(tf.one_hot(xrecord_onehot['p2'],3).numpy(), columns=['p2_0', 'p2_1', 'p2_2']), left_index=True, right_index=True).drop(['p2'], axis=1)
  xrecord_onehot = xrecord_onehot.merge(pd.DataFrame(tf.one_hot(xrecord_onehot['winner'],3).numpy(), columns=['winner_0', 'winner_1', 'winner_2']), left_index=True, right_index=True).drop(['winner'], axis=1)
  xrecord_onehot = xrecord_onehot.merge(pd.DataFrame(tf.one_hot(xrecord_onehot['model0'],3).numpy(), columns=['model0_0', 'model0_1', 'model0_2']), left_index=True, right_index=True).drop(['model0'], axis=1)
  xrecord_onehot = xrecord_onehot.merge(pd.DataFrame(tf.one_hot(xrecord_onehot['model1'],3).numpy(), columns=['model1_0', 'model1_1', 'model1_2']), left_index=True, right_index=True).drop(['model1'], axis=1)
  xrecord_onehot = xrecord_onehot.merge(pd.DataFrame(tf.one_hot(xrecord_onehot['model2'],3).numpy(), columns=['model2_0', 'model2_1', 'model2_2']), left_index=True, right_index=True).drop(['model2'], axis=1)
  xrecord_onehot = xrecord_onehot.merge(pd.DataFrame(tf.one_hot(xrecord_onehot['model3'],3).numpy(), columns=['model3_0', 'model3_1', 'model3_2']), left_index=True, right_index=True).drop(['model3'], axis=1)
  xrecord_onehot = xrecord_onehot.merge(pd.DataFrame(tf.one_hot(xrecord_onehot['model4'],3).numpy(), columns=['model4_0', 'model4_1', 'model4_2']), left_index=True, right_index=True).drop(['model4'], axis=1)
  xrecord_onehot = xrecord_onehot.merge(pd.DataFrame(tf.one_hot(xrecord_onehot['model5'],3).numpy(), columns=['model5_0', 'model5_1', 'model5_2']), left_index=True, right_index=True).drop(['model5'], axis=1)
  xrecord_onehot = xrecord_onehot.merge(pd.DataFrame(tf.one_hot(xrecord_onehot['model_choice'],6).numpy(), columns=['model_choice_0', 'model_choice_1', 'model_choice_2','model_choice_3','model_choice_4','model_choice_5']), left_index=True, right_index=True).drop(['model_choice'], axis=1)
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

def get_nn_Xy(record, length=5):
    """
    Builds an three dimensional array X that will input a 2 dimensional array for each value of y, which represents the player's next choice
    """

    xrecord_onehot = build_xrecord_onehot(record)
    print(xrecord_onehot)
    xrecord_onehot['valid_round'] = xrecord_onehot.index > length
    xrecord_onehot['valid_round'] = xrecord_onehot['valid_round'].astype(float)

    # builds X and y starting on round 8 based off of summary statistics of the previous 5 rounds
    X = np.ndarray((len(record)-2-length,length, xrecord_onehot.shape[1]))
    y = pd.Series(dtype=int)
    drop = []

    for i in range(length+1, len(record)-1):
      X[i-length-1] = xrecord_onehot[i-length:i].to_numpy()
      y = y.append(pd.Series({'index':xrecord_onehot['p1_next_choice'].iloc[i]}), ignore_index=True)
    
    this_round = X[-1]
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
    print(f'model0 len(record) = {len(record)}')
    if len(record) > 1:
        if len(record) > 5:
            vectors = []
            for i in range(1,5):
                vectors.append(vectorize(record['p1'].iloc[-i], record['p1'].iloc[-i-1]))
            print(f'vectors for model1 = {vectors}')
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
    """ Chooses the choice that would beat the player's least frequent recent choice. Combine and remake to choose least chosen option of last 3 """
    if len(record['p1']) > 4:
        least_freq = record['p1'].value_counts().iloc[-5:].index[-1]
        choice = beats(least_freq)
    elif len(record['p1']) > 0:
        least_freq = record['p1'].value_counts().index[-1]
        choice = beats(least_freq)
    else: choice = 0
    return int(choice)


def model4(record):
    """ builds a decision tree from record to predict player's next choice and returns the 
    choice to beat it"""
    if len(record) < 8:
        return int(0)

    X,y,this_round = get_nn_Xy(record)

    model = keras.models.load_model("nn_clf")
    guess = np.argmax(model.predict(this_round.reshape(1,5,39)))

    return beats(guess)


def model5(record):
    """ uses an sklearn model trained with historical data to predict next choice"""
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

def build_nn():    
    """ Builds the neural network classifier for model 4 from historical data """
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
    record = pd.read_csv('rps-record122.csv', index_col='Unnamed: 0')

    X,y,this_round = get_nn_Xy(record)

    model = keras.Sequential([
    keras.layers.Flatten(input_shape=(5, X.shape[2])),  # input layer (1)                      
    keras.layers.Dense(40, activation='elu', kernel_regularizer=keras.regularizers.l2(0.005)),
    keras.layers.Dense(20, activation='relu', kernel_regularizer=keras.regularizers.l2(0.005)),
    keras.layers.Dense(3, activation='softmax') ])

    model.compile(optimizer=keras.optimizers.Adamax(),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(X, y, epochs=300)

    model.save("nn_clf")

    print('nn file updated')
    
    return

#currently not in use
def always_changing(record): return len(record) > 5 and False not in [record['p1'].iloc[-i] != record['p1'].iloc[-i-1] for i in range(1,6)]

#need to be updated to priotize most recent rounds' scores over older rounds
def highest_score_model(model_scores): return int(model_scores.index(max(model_scores)))



### move this into its own file
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
        model_scores.append(score_model(4,record, drop_first=7)+0.2)
        model_scores.append(score_model(5,record, drop_first=5)+0.2)
        
    
    if len(record) < 1:
        model = 0
    elif record['winner'].iloc[-1] == 2:
        model = int(record['model_choice'].iloc[-1])
    else:
        model = highest_score_model(model_scores)
    
    print(f'Model Choices: {model_choices}')
    print(f'Model Scores: {model_scores}')
    print("Model {} chosen.".format(model))
    
    # next: build a ensembler that aggregates model suggestions weighted by their scores to choose
    
    choice = model_choices[model]
    
    return choice, model, model_choices
