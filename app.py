from flask import Flask, render_template, url_for, request, redirect
from src.model import computer_choice, build_historical_dtclf, build_nn
from src.game import *
from src.database import *
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///record.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# declare variables
rps_nums = ['Rock', 'Paper', 'Scissors']
global record
record = pd.DataFrame(columns = ['p1', 'p2', 'winner', 'model_choice', 'model0', 'model1', 'model2', 'model3', 'model4', 'model5', 'timestamp'])
global winning_score


#create SQLAlchemy object
class Todo(db.Model):
    n = db.Column(db.Integer, primary_key=True)
    p1 = db.Column(db.Integer)
    p2 = db.Column(db.Integer)
    winner = db.Column(db.Integer)
    model_choice = db.Column(db.Integer)
    model0 = db.Column(db.Integer)
    model1 = db.Column(db.Integer)
    model2 = db.Column(db.Integer)
    model3 = db.Column(db.Integer)
    model4 = db.Column(db.Integer)
    model5 = db.Column(db.Integer)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return '<Round %r>' % self.n


#define base app route asking for winning score choice
@app.route('/', methods=['POST', 'GET'])
def index():
    
    recs = Todo.query.order_by(Todo.n).all()
    for rec in recs:
            db.session.delete(rec)
            db.session.commit()
    
    

    if request.method == 'POST':
        winning_score = request.form['winning_score']

        #build_historical_dtclf()
        #build_nn()

        return render_template('play-studio.html', winning_score=winning_score, p_wins = 0, c_wins=0)
    else:
        winning_score = 10
        return render_template('index-studio.html', winning_score=winning_score)


#define app route during game
@app.route('/play/<int:winning_score>', methods=['POST', 'GET'])
def index1(winning_score, record=record):

    recs = Todo.query.order_by(Todo.n).all()
    ### needs to include timestamp
    for rec in recs:
        record = record.append({'p1':rec.p1, 'p2':rec.p2, 'winner':rec.winner, 'model_choice':rec.model_choice, 
                                'model0':rec.model0, 'model1':rec.model1, 'model2':rec.model2, 'model3':rec.model3, 
                                'model4':rec.model4, 'model5':rec.model5, 'timestamp':rec.timestamp}, ignore_index=True).astype(int, errors='ignore')

    try: p_wins = record['winner'].value_counts().loc[1]
    except: p_wins = 0
    try: c_wins = record['winner'].value_counts().loc[2]
    except: c_wins = 0 

    if request.method == 'POST':
        
        # play round
        print('\n'*2)

        # get round variables
        p2, model_choice, model_choices = computer_choice(record)
        p1 = int(request.form['choice'])
        winner = play_rps(p1,p2)

        #put round variables into db and df
        new_rec = Todo(p1=p1, p2=p2, winner=winner, model_choice=model_choice, model0=model_choices[0], model1=model_choices[1], 
                       model2=model_choices[2], model3=model_choices[3], model4=model_choices[4], model5=model_choices[5])
        
        try:
            db.session.add(new_rec)
            db.session.commit()
        except Exception:
            print("Error with adding rec:" + Exception)
        
        record = record.append({'p1':p1, 'p2':p2, 'winner':winner, 'model_choice':model_choice, 
                                'model0':model_choices[0], 'model1':model_choices[1], 'model2':model_choices[2], 
                                'model3':model_choices[3], 'model4':model_choices[4], 'model5':model_choices[5],
                                'timestamp':datetime.utcnow()}, ignore_index=True)

        # update win count
        if winner == 1:
            p_wins = p_wins + 1
        if winner == 2:
            c_wins = c_wins + 1
        
        recs = Todo.query.order_by(Todo.n).all()

        print(record)

        #play another round
        if p_wins < winning_score and c_wins < winning_score:
            return render_template('play-studio.html', recs=recs, winning_score=winning_score, rps_nums=rps_nums, p_wins=p_wins, c_wins=c_wins)

        #game over
        else:
            print('GAME OVER')
            
            #insert game record into persistent database
            table_name = 'record'

            try:game_id = query_to_df(f"SELECT MAX(game_id) FROM {table_name}").iloc[0,0] + 1
            except: game_id = 0
            record['game_id'] = game_id

            record['ip_address'] = request.remote_addr
            
            update_sql_from_df(record, table_name)

            return render_template('game-over-studio.html', recs=recs, winning_score=winning_score, rps_nums=rps_nums, p_wins=p_wins, c_wins=c_wins)

    # if the page is called by URL
    else:
        for rec in recs:
            db.session.delete(rec)
            db.session.commit()
        
        p_wins = 0
        c_wins = 0

        return render_template('play-studio.html', winning_score=winning_score, p_wins = 0, c_wins=0)

if __name__ == "__main__":
    app.run(debug=True)