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



from game import Agent

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
    enemies = [(i, successor.getAgentState(i)) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a[1].isPacman and a[1].getPosition() != None]
    features['numInvaders'] = len(invaders)


    # make agent go to openings we are defending
    features['middleOpening'] = self.getMazeDistance(myPos, self.middleOpening)

    

    if len(invaders) > 0:
      #record that the agents are offensive
      for inv in invaders: self.offOpps[inv[0]] += 1

      dists = [self.getMazeDistance(myPos, a[1].getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)
      
      #stop caring about middle when there are invading pacmans
      features['middleOpening'] = 0

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1


    # punish defense for being pacman/on opposite side
    if me.isPacman: features['remainGhost'] = 1

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
            'middleOpening': -1,
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
            'avoidDefY': 1}




  def final(self, gameState):
    self.observationHistory= []
    #self.recordResults(gameState)




def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'OffensiveReflexAgent'):
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

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = betterEvaluationFunction
    self.treeDepth = int(depth)


class AlphaBetaAgent(MultiAgentSearchAgent, CaptureAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def a_minVal(self, gameState, ghostNo, depth, alpha, beta, maxDepth):
    #print "called min val at depth ", depth

    #if this is the last ghost, call max for pacman 
    if ghostNo == gameState.getNumAgents()-1:
      best_min1 = float('inf')
      for act in gameState.getLegalActions(ghostNo):
        nextState = gameState.generateSuccessor(ghostNo, act)
        v = self.a_maxVal(nextState, depth+1, alpha, beta, maxDepth)

        # v = min(v, max)
        if v < best_min1: 
          best_min1 = v

        if best_min1 <= alpha: return best_min1
        beta = min(beta, best_min1)

      return best_min1

    #otherwise call for next ghost
    else:
      best_min2 = float('inf')
      for act in gameState.getLegalActions(ghostNo):
        nextState = gameState.generateSuccessor(ghostNo, act)
        v = self.a_minVal(nextState, ghostNo+1, depth, alpha, beta, maxDepth)
       
        # v = min(v, min)
        if v < best_min2:
          best_min2 = v

        if best_min2 <= alpha: return best_min2
        beta = min(beta, best_min2)


      return best_min2


  def a_maxVal(self, gameState, depth, alpha, beta, maxDepth): #max is pacman agent

    if depth > 4:
      return self.evaluationFunction(gameState)


    best_max = float('-inf')
    for act in gameState.getLegalActions(0):
      nextState = gameState.generateSuccessor(0, act)
      v = self.a_minVal(nextState, 1, depth, alpha, beta, maxDepth)

      #v = max(v, min)
      if v > best_max: 
        best_max = v

      if best_max >= beta: return best_max #if we already know it's a bad route, return
      alpha = max(alpha, best_max)

    return best_max


  def alpha_beta(self, gameState, maxDepth):
    best = (None, float('-inf'))
    alpha = float('-inf')
    beta = float('inf')
    for act in gameState.getLegalActions(0):
      nextState = gameState.generateSuccessor(0, act)
      v = self.a_minVal(nextState, 1, 0, alpha, beta, maxDepth)

      #v = max(minVal for actions)
      if v > best[1] and not act == 'Stop':
        best = (act, v)

    return best[0]


    
  def getAction(self, gameState):
    """
      Returns the minimax action using self.treeDepth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    maxDepth = 2
    return self.alpha_beta(gameState, maxDepth)

  def chooseAction(self, gameState):
  	return self.getAction(gameState, 1)






def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>

    The evaluation function uses 3 different metrics to evaulate the game state

    The first metric is the distance to the closest food peice. The distance is found
    using a bfs search, and it's weight towards the evaluation function's value is 1/106

    The second metric is the number of food peices left. It's weight is 5/106

    The last metric is the distance to the closeset ghost. Most of the time, pacman
    ignores the ghosts if they are further than 2 squares away. Otherwise, pacman 
    really freaks out and really doesn't want to be within 2 squares of a ghost

    All the metrics are added together (or subtracted in the case of the distance 
    and number of food)

  """


  SCORE_WEIGHT = 5
  FOOD_DIST_WEIGHT = 1
  GHOST_WEIGHT = 100

  #useful starting info
  newx, newy = currentGameState.getPacmanPosition()
  foodList = currentGameState.getFood().asList()
  newGhostStates = currentGameState.getGhostStates()
  newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
  num_food = len(foodList)

  walls = currentGameState.getWalls()

  if currentGameState.isWin(): 
    return float('inf')

  # find the closest food
  dist = 1

  if not num_food == 0:
    import util
    from sets import Set 

    q = util.Queue()
    visited = Set()
    foodSet = Set(foodList)

    f = (currentGameState.getPacmanPosition(), 0)

    q.push(f) #push (coords, distance)
    while not q.isEmpty():

      (currCoords, d) = q.pop()
      visited.add(currCoords)

      if currCoords in foodSet:
        dist = d
        break

      for (dx, dy) in [(1,0), (-1, 0), (0,1), (0, -1)]:
        (nextX, nextY) = (currCoords[0] + dx, currCoords[1] + dy)
        if not (nextX, nextY) in visited and not walls[nextX][nextY]:
          t = ((nextX, nextY), d+1)
          q.push(t)


  #take reciprocal because we want closer to be better
  food_component = -dist * FOOD_DIST_WEIGHT

  # if pacman thinks he'll hit a ghost, give state a very low rating
  hitGhost = 0
  for ghost in newGhostStates:
    gx, gy, = ghost.getPosition()
    if abs(newx - gx) + abs(newy - gy) < 2:
      hitGhost = -1

  ghost_component = hitGhost * GHOST_WEIGHT

  r = ghost_component - ((len(foodList) * SCORE_WEIGHT)) + food_component + 1000 #add a bias term to make score positive
  return r

