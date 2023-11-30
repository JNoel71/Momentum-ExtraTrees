import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import log_loss, accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import ParameterGrid

def removeEarlyGames(df,games=20):
    """Remove a number of early games for each team from the dataframe.
    
    Parameters:
        df - the unedited dataframe.
        games - the number of games to be removed.
        
    Returns:
        df - the original dataframe now with the desired number of games removed.
    """
    #get all team names
    teams = df.Home_Team.unique()

    #get all seasons
    seasons = df.season.unique()

    #where the games that will be removed are stored
    gamesToRemove = set()

    for i in seasons:
        #get the games from a given season
        season = df[df["season"] == i].sort_values("Game_Id")
        for j in teams:
            #get the teams first x games
            earlyGames = season[(season["Away_Team"] == j) | (season["Home_Team"] == j)].head(games)
            gameList = earlyGames.Game_Id.unique()
            gamesToRemove.update(gameList)
    
    #remove games from the dataframe
    df = df[~df.Game_Id.isin(list(gamesToRemove))]

    return df

def testPlayoffSeason(classifier,df,season):
    """Test a given playoff season.
    
    Parameters:
        classifier - the model to be fit.
        df - the dataframe with the required data.
        season - the season to be tested.
    """
    #create training
    trainingFrame = df[(df['season'] < 2020)] #train on data before 2020
    trainingFrame = trainingFrame[trainingFrame['isPlayoff'] == 1] #only use playoff games
    trainingFrame = removeEarlyGames(trainingFrame,3) #remove each teams first 3 games of each playoff season
    trainingFrame = trainingFrame[trainingFrame['RegOrOT'] != 'OT'] #remove overtime games
    trainingFrame = trainingFrame.drop(['Game_Id','RegOrOT','Away_Team','Home_Team','season','isPlayoff'], axis=1) #remove unneeded columns
    #split X and y
    trainingX = trainingFrame[trainingFrame.columns.difference(['Outcome'])]
    trainingY = trainingFrame['Outcome'].astype('int32')

    #create testing
    testingFrame = df[df['season'] == season] #test on selected season
    testingFrame = testingFrame[testingFrame['isPlayoff'] == 1] #only use playoff games
    testingFrame = testingFrame.drop(['Game_Id','RegOrOT','Away_Team','Home_Team','season','isPlayoff'], axis=1) #remove unneeded columns
    #split X and y
    testingX = testingFrame[testingFrame.columns.difference(['Outcome'])]
    testingY = testingFrame['Outcome'].astype('int32')

    #select features
    selector = SelectFromModel(estimator=classifier,threshold='mean').fit(trainingX,trainingY)
    trainingX = trainingX[trainingX.columns[selector.get_support(indices=True)]]
    testingX = testingX[testingX.columns[selector.get_support(indices=True)]]
    
    #score the cross validated training models
    skf = StratifiedKFold(n_splits=10)
    LogLoss = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
    scoresLL = cross_val_score(classifier,trainingX,trainingY,cv=skf,scoring=LogLoss) 
    print('Cross Validated Log Loss: ' + str(scoresLL.mean()))
    scoresAcc = cross_val_score(classifier,trainingX,trainingY,cv=skf,scoring='accuracy') 
    print('Cross Validated Accuracy: ' + str(scoresAcc.mean()))
    
    #fit the model to the training data and predict testing data
    classifier.fit(trainingX,trainingY)
    preds = classifier.predict(testingX)
    proba = classifier.predict_proba(testingX)

    #get baseline values
    baseline = trainingY.sum()/trainingX.shape[0]
    prob = []
    for i in range(testingY.shape[0]):
        prob.append(baseline)
    testingBaselineAcc = accuracy_score(testingY,[round(p) for p in prob])
    testingBaselineLL = log_loss(testingY,prob)
    
    #output model performance
    print('log Loss: ' + str(log_loss(testingY,proba)))
    print('Accuracy: ' + str(accuracy_score(testingY,preds)))
    print('Total Increase: ' + str((accuracy_score(testingY,preds)/testingBaselineAcc) - ((log_loss(testingY,proba))/testingBaselineLL)))
    print("")

def customGS(model,df,season):
    """Custom grid search.
    
    Parameters:
        model: ExtraTreesClassifier to be fit to the data.
        df - the dataframe to be split for training and testing.
        season - the season to split training and testing on.
    """
    #get seasons less than given season for training
    trainingFrame = df[df['season'] < season]

    #only use playoffs, remove first 3 games for each team, remove overtime games, drop unneeded columns
    trainingFrame = trainingFrame[trainingFrame['isPlayoff'] == 1]
    trainingFrame = removeEarlyGames(trainingFrame,3)
    trainingFrame = trainingFrame[trainingFrame['RegOrOT'] != 'OT']
    trainingFrame = trainingFrame.drop(['Game_Id','RegOrOT','Away_Team','Home_Team','season','isPlayoff'], axis=1)

    #split X and y 
    trainingX = trainingFrame[trainingFrame.columns.difference(['Outcome'])]
    trainingY = trainingFrame['Outcome'].astype('int32')

    #determine baseline accuracy
    baseline = trainingY.sum()/trainingX.shape[0]
    print('Baseline Acc: ' + str(baseline))

    #create data for baseline log loss calculation
    prob = []
    for i in range(trainingY.shape[0]):
        prob.append(baseline)

    #get baseline log loss
    baselineLL = log_loss(trainingY,prob)
    print('Baseline LL: ' + str(baselineLL))

    ll = [] #keep track of log loss
    acc = [] #keep track of accuracy
    total = [] #keep track of predictive power (formula 1)
    p = [] #keep track of parameters

    #parameter grid to be searched
    params = {
    'max_depth':[None, 4, 8, 16],
    'min_samples_split': [2, 4, 8, 16],
    'min_samples_leaf': [1,10,20,30],
    'max_features': ['sqrt','log2'],
    'n_estimators': [150,250,500,750],
    'random_state':[0]
    }
    
    #create the parameter grid
    param_grid = ParameterGrid(params)

    #create cross validation folds
    skf = StratifiedKFold(n_splits=10)

    #create log loss scorer
    LogLoss = make_scorer(log_loss, greater_is_better=False, needs_proba=True)

    #iterate through all possible sets of parameters
    for dict_ in param_grid:
        #fit model
        model = ExtraTreesClassifier(**dict_)

        #select features
        selector = SelectFromModel(estimator=model,threshold='mean').fit(trainingX,trainingY)

        #score the model
        scoresLL = cross_val_score(model,selector.transform(trainingX),trainingY,cv=skf,scoring=LogLoss)
        scoresAcc = cross_val_score(model,selector.transform(trainingX),trainingY,cv=skf,scoring='accuracy')
        acc.append(scoresAcc.mean())
        ll.append(scoresLL.mean())
        total.append((scoresAcc.mean()/baseline) - ((-scoresLL.mean())/baselineLL))
        p.append(dict_)

    #get model which performed best per formula 1
    index = total.index(max(total))
    print("Best Score LL: " + str(ll[index]))
    print("Best Score ACC: " + str(acc[index]))
    print("Best Score Total: " + str(total[index]))
    print("Parameters: ")
    print(p[index])
    print("")

def testCombinations():
    """Test all possible combinations of one frequency-based and momentum-based feature set using the custom
    grid search."""
    Mo = [3,4,5,6]
    Fr = [3,4,5,6]
    
    #perform custom grid search on all possible combinations of feature sets
    for i in Mo:
        for j in Fr:
            momentumDF = pd.read_csv("DataFrames/momentum" + str(i) + "NoCross.csv")
            frequencyDF = pd.read_csv("DataFrames/frequency" + str(j) + "NoCross.csv")
            combinedDF = pd.merge(momentumDF, frequencyDF, on=['Game_Id','RegOrOT','Away_Team','Home_Team','season','isPlayoff','Outcome'], how='inner')

            #perform grid search
            print("ET Combined: " + str(i) + " " + str(j))
            customGS(ExtraTreesClassifier(random_state=0),combinedDF,2020)

   
def mainTest():
    """Test used to compare models after the best hyperparameters and game intervals were discovered."""

    #Best parameters
    combinedParams = {'max_depth': 16, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 16, 'n_estimators': 250, 'random_state': 0}
    freqParams = {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 16, 'n_estimators': 250, 'random_state': 0}
    moParams = {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 8, 'n_estimators': 500, 'random_state': 0}

    #three separate models to be used
    classifierCombined = ExtraTreesClassifier(**combinedParams)
    classifierFreq = ExtraTreesClassifier(**freqParams)
    classifierMo = ExtraTreesClassifier(**moParams)

    #three separate feature sets
    momentumDF = pd.read_csv("DataFrames/momentum" + str(4) + "NoCross.csv")
    frequencyDF = pd.read_csv("DataFrames/frequency"+ str(3) + "NoCross.csv")
    combinedDF = pd.merge(momentumDF, frequencyDF, on=['Game_Id','RegOrOT','Away_Team','Home_Team','season','isPlayoff','Outcome'], how='inner')

    print("2021")
    testPlayoffSeason(classifierCombined,combinedDF,2021)
    testPlayoffSeason(classifierFreq,frequencyDF,2021)
    testPlayoffSeason(classifierMo,momentumDF,2021)
    print("")
    print("2020")
    testPlayoffSeason(classifierCombined,combinedDF,2020)
    testPlayoffSeason(classifierFreq,frequencyDF,2020)
    testPlayoffSeason(classifierMo,momentumDF,2020)
    print("")
 
#mainTest()


