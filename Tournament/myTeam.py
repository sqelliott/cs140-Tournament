# myTeam.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import *
import datetime
import math

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'Agent1', second = 'Agent1'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class Agent1(CaptureAgent):

  def __init__(self,index, timeForComputing =.1):
    CaptureAgent.__init__(self,index, timeForComputing)
    self.epsilon = 0.05
    self.weights = util.Counter()
    self.alpha = 0.5
    self.discountRate = .01
    self.training = True 
    tmp = self.maintainedWeights()
    for t in tmp:
      self.weights[t] = tmp[t]
    

    if not self.training:
      self.epsilon = 0




  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor



  "functions to user for learning"
  def getQValue(self, state, action):
    qValue = self.weights * self.getFeatures(state,action)
    return qValue

  def getValue(self,state):
    return self.getQValue(state,self.getPolicy(state))


  def getPolicy(self,state):
    actions = state.getLegalActions(self.index)
    actionValues = [ (self.getQValue(state,action),action) for action in actions]
    best = max(actionValues)
    return best[1]


  def getAction(self,state):
    actions = state.getLegalActions(self.index)
    action = self.getPolicy(state)

    if util.flipCoin(self.epsilon): 
      action =  random.choice(actions)

    if self.training: 
      nextState = self.getSuccessor(state,action)
      self.update(state,action,nextState)
    return action


  def update(self,state,action,nextState):
    features = self.getFeatures(state,action)
    for feature in features:
      reward = self.getScore(state) - self.getScore(nextState) - 1 # negative living reward
      correction = (reward) + self.discountRate * self.getValue(nextState) - self.getQValue(state,action)
      self.weights[feature] += self.alpha * correction * features[feature]


  "user this to keep track of training"
  def final(self, gameState):
    self.observationHistory = []
    self.recordWeights()

  "update for better features"
  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)

    foodList = self.getFood(successor).asList()
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance

    "promote distance for offense"
    # import pdb;pdb.set_trace()
    # team = self.getTeam(successor)
    # for t in team:
    #   if t != self.index: b = t
    # teamPos = successor.getAgentState(b).getPosition()
    # d = self.getMazeDistance(myPos,teamPos)
    # features['off_team_dist'] = d

    return features

  def maintainedWeights(self):
    return {'successorScore': 0, 'distanceToFood' : 0}

  def recordWeights(self):
    f = open ('weightRecords.txt', 'a')
    now = datetime.datetime.now()

    f.write(now.strftime("%m-%d %H:%M")+" ")
    for w in self.weights:
      f.write("{ "+w+":"+str(self.weights[w])+" } ")
    f.write("\n")








class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on). 
    
    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    ''' 
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py. 
    '''
    CaptureAgent.registerInitialState(self, gameState)

    ''' 
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    ''' 
    You should change this in your own agent.
    '''

    return random.choice(actions)

