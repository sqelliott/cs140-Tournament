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



#################
# Team creation #
#################


class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
  def registerInitialState(self,gameState):
    CaptureAgent.registerInitialState(self,gameState)
    self.index2 = (self.index + 2) % 4
    if self.red: side = -2
    else: side =2 

    #init which opponents are guessed to be offensive agents
    self.offOpps = util.Counter()
    self.offOpps[(self.index+1)%4]=0
    self.offOpps[(self.index+3)%4]=0
    self.defensiveOpenings = []


    walls = gameState.getWalls()

    gameBoardHeight= walls.height
    gameBoardWidth = walls.width

    centerLine = gameBoardWidth/2


    #find the opening closest to the middle
    self.middleOpening = (centerLine, 0)


    for i in range(1, gameBoardHeight-1):
      if not walls[centerLine][i] and not walls[centerLine + side][i]:
        self.defensiveOpenings.append((centerLine + side, i))

    dist = []
    for (doX, doY) in self.defensiveOpenings:
      d = abs(gameBoardHeight/2 - doY)
      dist.append((d, (doX, doY)))

    self.middleOpening = min(dist)[1]

   

    self.capsules = gameState.getCapsules()

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

    features['middleOpening'] = self.getMazeDistance(myPos, self.middleOpening)

    # punish defense for being pacman/on opposite side
    if a.isPacman: features['remainGhost'] = 1

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
    return {'numInvaders': -200, 
            'onDefense': 000, 
            'invaderDistance': -100, 
            'defensiveOpenings': -1,
            'middleOpening': -5,
            'stop': -10, 
            'reverse': -00,
            'teamAttackdist': 1,
            'remainGhost': -2}

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

  def registerInitialState(self,gameState):
    ReflexCaptureAgent.registerInitialState(self,gameState)
    self.defOpps = util.Counter()
    self.defOpps[(self.index+1)%4]=0
    self.defOpps[(self.index+3)%4]=0

  def getFeatures(self, gameState, action):
    # score feature
    features  = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    me        = successor.getAgentState(self.index)
    myPos     = me.getPosition()
    "info to get teammate information"
    teammate  = successor.getAgentState(self.index2)
    teammPos  = teammate.getPosition()


    # Compute distance to the nearest food
    foodList = self.getFood(successor).asList()
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance

    # avoid an enemy ghost while on enemy side
    enemies = []
    for i in self.getOpponents(successor):
      a = successor.getAgentState(i)
      enemies.append(a)
      if not a.isPacman and a.getPosition() != None:
        self.defOpps[i] += 1
        #print i,self.defOpps[i]

    # make offense agent avoid y-axis of predicted defense agent
    key = None
    i   = None
    for k in self.defOpps:
      if key == None:
        key = k
        i = self.defOpps[k]
      elif i > self.defOpps[k]:
        key = k
        i = self.defOpps[k]
    if successor.getAgentState(key).getPosition() != None and successor.getAgentState(key).getPosition() !=None:
      iPos    = successor.getAgentState(key).getPosition()
      absDiff = abs(myPos[1] - iPos[1])
      features['avoidDefY'] = absDiff





    ghosts    = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    num_ghost = len(ghosts)


    if num_ghost > 0:
      dist = [(self.getMazeDistance(myPos,a.getPosition()), a) for a in ghosts]
      if min(dist) < 7 and a.scaredTimer<2:
        features['ghostDist'] = min(dist)
      else:
        features['ghostDist'] = 0


    # if both agents are pacman, make them 
    # stay away each other
    if me.isPacman and teammate.isPacman:
      t = teammate.getPosition()
      if self.getMazeDistance(myPos, t) <5:
        features['teamAttackDist'] = self.getMazeDistance(myPos,t)


    #capsule feature to get close if close by
    cap_dist = [self.getMazeDistance(myPos,a) for a in self.capsules]
    if len(cap_dist)>0:
      if min(cap_dist) < 8:
        features['capsule'] = min(cap_dist)

    # eat opponent pacman while in ghost state
    if not me.isPacman:
      if num_ghost>0: 
        if min(dist) <4:
          features['offGhostState'] = 1




    


    # promote not stoping
    # should have feature that promotes exploring different areas
    # after having a high distrution in an area
    if action == Directions.STOP: features['stop'] = 1


    
  
    return features

  def getWeights(self):
    return {'successorScore': 100, 
            'distanceToFood': -5, 
            'ghostDist': 15,
            'teamAttackDist':5,
            'stop': -100,
            'cap_dist':2,
            'offGhostState':10,
            'avoidDefY': 7}




  def final(self, gameState):
    self.observationHistory= []
    #self.recordResults(gameState)




def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
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
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

