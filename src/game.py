import pandas as pd
import numpy as np

rps_nums = ['rock', 'paper', 'scissors']

def beats(n):
    """ Return the choice that will beat the given choice """
    if int(n) == 2:
        return int(0)
    else:
        return int(n)+1
    
    
def loses_to(n):
    """ Return the choice that will lose to the given choice """

    if int(n) == 0:
        return int(2)
    else: 
        return int(n)-1


def play_rps(p1, p2, printed=True):
    """ Plays a round with the two scores and returns the winner """
    
    if printed == True:
        print(f'You played {rps_nums[p1].upper()} and the computer played {rps_nums[p2].upper()}')
    
    if p1 == beats(p2):
        winner = 1
    elif p1 == loses_to(p2):
        winner = 2
    else:
        winner = 0
    
    if printed == True:
        if winner == 0:
            print('Tie game!', end='\n')
        if winner == 1:
            print(f'You win!'.upper(), end='\n')
        if winner == 2:
            print(f'The computer wins!'.upper(), end='\n')
        
    return winner
