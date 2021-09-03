import pickle


def openAbilityScores():
    with open('playerScores.txt', 'rb') as f:
        log = pickle.load(f)
    f.close()
    print(log)
    return(log)