# Rock Paper Scissors AI
A full stack python Flask artificial intelligence project capable of beating the human user over 60% of the time using a custom scoring system to ensemble six models (naïve logic-based, decision tree, neural network) trained on both game-level and stored historical data in AWS RDS Cloud SQL database.

[Play the game here](https://rps-ai-game.herokuapp.com/)

![app_interface](static/images/rps-webapp-screenshot.jpg)


# Overview
Although at first glance, Rock Paper Scissors might seem like a low-level game, I actually came to think of it in the opposite way when conceiving this AI. These days, it is easy to assume that a computer can beat you in chess, because it can harness all of its computing power to see all possible outcomes and choose the ones that benefit it. Rock Paper Scissors, on the other hand, is a game that seems impossible to be good at. In theory, decisions can be made at random and have no less likelihood of winning than a well thought-through decision. My theory though, was that humans can’t actually make random decisions, and that if an AI could learn to understand the ways in which humans make their choices, even if the human is trying to do so in a random pattern, then the AI would be able to significantly exceed 33% accuracy in guessing the player’s decisions. 

I started out by simply hard coding the different ways that I could think of that humans would make decisions: choosing the same thing over and over, choosing in a pattern, or trying to make the choice that they hadn’t used in a while. I built models that would predict the player’s next choice if they were using any of these methods, and then used logic-based criteria to try and decide which model fit the player’s behavior based on a record of the previous rounds. This was the first stage of the project, ran in a jupyter notebook, and initially played pretty well. It would fall into certain patterns easily, however, and could be reliably tricked by a savvy player. 


At this point, I realized that there were a lot of improvements that I could make and got excited to flesh the project out more. I built a Flask webapp and hosted it on Heroku so that I could share it with friends. I then built a cloud database on AWS to capture the data from every time that it was played, knowing that this data could give me the power to build much more sophisticated models.

I began a process of analysing the performance of my models and tweaking them. I also replaced the simple logic-based model selection process with a new scoring system to make the decision of which model would be used to make the next round’s choice, which I’ll go into more detail about below. I created a mobile-friendly app using bootstrap, improved the design for a more engaging user experience, and then sent the link to my network to play against and collect data from.

With this data, I began to implement machine learning models alongside the naive logic-based ones. I created two decision trees, one that trained and predicted only based on game-level data, and one that trained at the start of the game on the full historical data set to make its predictions. At this point, progress slowed as the app’s scope increased, and I had to work through package dependency, data quality, backend, and model exportation issues. However, I continued to iteratively update the app functionality in response to data analysis I was performing and new ideas I continued to have to improve it. In most recent changes, I’ve added a neural network model to replace the locally-trained decision tree which was performing poorly, and updated the scoring system to its current exponential state in order to prioritize recent rounds and ensure unpredictability. These days, even when I play against the AI, I have a hard time beating it. 
 

# Stack
The app is written in Python with Flask, is hosted on Heroku, reads and writes to an AWS RDS hosted PostgreSQL database, and the UI was written with Bootstrap HTML. Data is managed in pandas, and the ML models were built and tested with scikit-learn and Tensorflow. 

In its current MVP version, the app can only support one player at a time.



# AI
### Ensembling
Every round, the computer_choice function chooses which model it will use as the AI’s choice for the coming round. This is done by scoring the performance of each model given the current game record. 

![scoring](https://imgur.com/oPYXR5a.jpg)

This scoring system is exponential in order to prioritize recent model performance, making it responsive to strategies that a player might be using. Firstly, this allows the model to overcome simple patterns that a player may be using, such as playing the same choice constantly or switching choices in a repeated pattern. Secondly, it ensures that the same model is not consistently repeated, which would allow for the player to figure out how it worked and beat it. If the player did figure out a model, it would lose and quickly earn a negative score due to the priority that exponential scoring places on the most recent rounds. And finally, it allows the models that may more accurately predict the player’s thinking to be used more frequently. The decision tree and neural network models are given a +0.15 boost to their scores in order to give these sophisticated models precedence in the decision making process. 

![backend](https://imgur.com/JHbpxPj.jpg)


### Models

Each of the models takes a different approach to understanding how a player will make their next choice. The first four models make very specific assumptions about the way a player is making choices, and thus are only applicable in the context of a player acting in that given way.

Model 0: Chooses the choice that would lose to or beat the player's previous choice. Based on the assumption that a player will continue to either not repeat or to repeat previous choices.

Model 1: Vector-based choice based on past three rounds. Based on the assumption that a player will be using a certain pattern (ex. Rock, Paper, Scissors, Rock, Paper, Scissors, etc.)

Model 2: Chooses the choice that would beat the player's most frequent recent choice. Based on the assumption that a player has a certain choice that they keep making.

Model 3: Chooses the choice that would beat the player's least frequent recent choice. Based on the assumption that the player will try to play the choice they haven’t played in a while.

With the ensembler built to favor models that are predicting accurately in recent rounds, these models will eventually rise in score to become the deciding model in the computer’s choice if the human player repeats the behavior on which their assumption is built enough times. 


The last two models are more sophisticated, one being a decision tree and the other being a neural network, both trained on the historical dataset of games played against the AI. These models take longer to kick in to the AI’s decision making process because they need to see at least 7 rounds and 5 rounds of player data respectively to feed into their ML models as input. These models are made aware not only of the player’s choices from previous rounds, but also the computer’s choices, who the winner was, which model the AI used to make its choices, and each individual model’s choices for the previous rounds.

Model 4: Uses a pickled scikit-learn neural network model trained on historical data to predict and beat the player’s next choice based on data from the previous 7 rounds.

Model 5: Uses a pickled scikit-learn decision tree model trained on historical data to predict and beat the player’s next choice based on data from the previous 5 rounds.

The decision tree and neural network models are given a +0.15 boost to their scores once their necessary number of rounds have been reached and they have been added into the ensembler’s model options. This gives these sophisticated models precedence in the AI’s decision making process, as they are far more aware of the nuances of player behavior than the other models. Nonetheless, if a player is consistently displaying a behavior pattern that is accounted for by one of the naive models, that model will still overcome the small 0.15 point disadvantage and kick in to beat the player. 

Beginning the game using the naive models helps to solve the cold start problem, where the AI knows nothing yet about the player who it is starting a new game against. (The only use of a random number generator is for the very first round, as it is necessary to make the AI not entirely predictable, otherwise random numbers were entirely avoided in the building of this app.) The initial AI is therefore doing the best it can by its logic-based models until it gets a better understanding of the player and can turn on its more intelligent models. 

### Ensembling Model Selection Visualization

![model_selection](https://i.imgur.com/2txVzKz.png)


### Statistics

My goal was to have the AI win over 55% of the time. Currently, the AI’s win percentage sits at 61.8%. 
![win margins](https://imgur.com/HwdHglR.png)

To analyze how the models perform, we will look at two datasets: one that shows the result of the model’s choice in every round against what the player actually chose that round, and the next showing how the models actually performed against the player’s choice on rounds that the ensembler chose that model. As can be seen below, the top performing model on all rounds (judging by win / loss ratio) is Model 4, which has made the winning choice 203 times and the losing choice 168 times. The second best is the decision tree model at Model 5, which has made the winning choice 273 times and the losing choice 223 times. Note: these model scores are only calculated for the most recent iteration of the model, thus the lower total counts for the neural network at model 4. 
![full model performance](https://imgur.com/f9EaFVI.png)

Looking only at the rounds in which models were chosen by the ensembler, the top performing model is also the neural network Model 4 with 28 wins and 16 losses, followed closely by Model 2 with 59 wins and 37 losses. The naive models 0-2 can be seen to have a much higher win/loss ratio when they are chosen than they do across all rounds, which means that the ensemble is choosing the correct times to play the models in general. Comparing the two figures, the ensembler seems to use Model 2 especially effectively, while Model 3 actually experiences a much lower win/loss ratio when it is chosen by the ensembler, meaning that it is not being selected at the correct times.
![chosen model performance](https://imgur.com/cy5bAeu.png)

### Takeaways

This AI has been able to outperform my expectations by having two important characteristics. Firstly, it is capable of beating a player with its naive models when they fall into a simple pattern, and by its complex models when they might be following a more complex, but still measurable, decision making process. And secondly, it is itself hard to predict. As the person who programmed it, I still cannot predict what its choices will be when I am playing against it; a fact that came to be especially after introducing the machine learning models. Despite the fact that there is no randomness whatsoever in its decision making after the first round, it is nonetheless sufficiently beyond the player’s ability to calculate what the AI will choose.

The ensemble of models that powers this AI is an example of a framework that can be used for AI decision making in real time where data bear weak correlation to each other and most recent trends need to be prioritized over averages of the whole dataset.


### Future ideas: 

Some ideas to improve the AI: 
- Hosting competitions to build a larger dataset for training and validation of new ideas
- Test an ensembler using a weighted average of model choices given their scores
- Cluster player behavior into categories (ex. calculating, risky, random, etc.) and then train ML models to each cluster
- Build a meta layer above the ensembler that learns to enact multi-round strategies through reinforcement learning

Another possibility is to turn the app into a platform where data scientists or students can compete in building Game AI and trying to beat their peers’ AI and move up a leaderboard.



