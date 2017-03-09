# reflexAgent.py

from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import *
import datetime


class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
  def registerInitialState(self,gameState):
    CaptureAgent.registerInitialState(self,gameState)
    self.index2 = (self.index + 2) % 4
    self.capsules = gameState.getCapsules()
    

    #init which opponents are guessed to be offensive agents
    self.offensiveOpponenets = []

    self.defensiveOpenings = []

    walls = gamestate.getWalls()

    gameBoardHeight= len(walls[0])
    gameBoardWidth = len(walls)

    centerLine = gameBoardWidth/2

    for i in range(0, gameBoardHeight):
      if walls[centerLine][i]:
        self.defensiveOpenings.append((centerLine, i))


    


  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    return random.choice(bestActions)

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

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights()
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}
