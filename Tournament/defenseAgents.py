# defenseAgents.py

from reflexAgent import ReflexCaptureAgent
import random, time, util
from game import Directions
import game
from util import *
import datetime

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    me = successor.getAgentState(self.index)
    myPos = me.getPosition()

    "info to get teammate information"
    teammate  = successor.getAgentState(self.index2)
    teammPos  = teammate.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if me.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    # make agent go to capsules we are defending
    dOpenings = [self.getMazeDistance(myPos,q) for q in self.defensiveOpenings]
    if len(dOpenings) > 0:
      features['defensiveOpenings'] = min (dOpenings)


    # team distance maintanence
    t = teammate.getPosition()
    if self.getMazeDistance(myPos, t) <2:
      features['teamAttackDist'] = self.getMazeDistance(myPos,t)

    # make a feature to send pacman to a good start location
    # middle of map at start



    return features

  def getWeights(self):
    return {'numInvaders': -100, 
            'onDefense': 000, 
            'invaderDistance': -10, 
            'defensiveOpenings': -3
            'stop': -100, 
            'reverse': -20,
            'defensiveOpenings': -3,
            'teamAttackdist': 1}
