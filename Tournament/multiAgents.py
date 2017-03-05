# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """


  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    SCORE_WEIGHT = 100
    FOOD_DIST_WEIGHT = 100
    GHOST_WEIGHT = 500

    '''
    Evaluation function has a ghost component, a food distance component, and a food number
    component. The most important is ghosts as pacman is very afraid of getting too close. 
    The other two components are of equal value, and are 1/5 as important as the ghost componenet
    '''

    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newx, newy = newPosition = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # find the closest food
    dist = 999
    newFood = successorGameState.getFood()
    if not len(newFood.asList()) == len(oldFood.asList()): 
      dist = 0

    else:
      for (foodx, foody) in newFood.asList():
        d = abs(newx - foodx) + abs(newy - foody)
        if d < dist: dist = d


    #take reciprocal because we want closer to be better
    food_component = -dist * FOOD_DIST_WEIGHT
    hitGhost = 0
    for ghost in newGhostStates:
      gx, gy, = ghost.getPosition()
      isScared = ghost.scaredTimer
      if abs(newx - gx) + abs(newy - gy) < 2:
        hitGhost = -1

    ghost_component = hitGhost * GHOST_WEIGHT


    "*** YOUR CODE HERE ***"
    r = ghost_component + food_component - (len(newFood.asList()) * SCORE_WEIGHT)
    return r

def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

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
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.treeDepth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """


  def a_minVal(self, gameState, ghostNo, depth):
    #print "called min val at depth ", depth

    #if this is the last ghost, call max for pacman 
    if ghostNo == gameState.getNumAgents()-1:
      best = 999999
      for act in gameState.getLegalActions(ghostNo):
        nextState = gameState.generateSuccessor(ghostNo, act)
        v = self.a_maxVal(nextState, depth+1)
        if v < best: 
          best = v

      return best

    #otherwise call for next ghost
    else:
      best = 999999
      for act in gameState.getLegalActions(ghostNo):
        nextState = gameState.generateSuccessor(ghostNo, act)
        v = self.a_minVal(nextState, ghostNo+1, depth)
        if v < best:
          best = v

      return best


  def a_maxVal(self, gameState, depth): #max is pacman agent

    if depth == self.treeDepth:
      return self.evaluationFunction(gameState)


    best = -999999
    for act in gameState.getLegalActions(0):
      nextState = gameState.generateSuccessor(0, act)
      v = self.a_minVal(nextState, 1, depth) #trying to maximize returns from ghosts
      if v > best: 
        best = v

    return best


  def a_minMax(self, gameState):
    best = (None, -999999)
    for act in gameState.getLegalActions(0):
      nextState = gameState.generateSuccessor(0, act)
      v = self.a_minVal(nextState, 1, 0)
      if v > best[1] and not act == 'Stop':
        best = (act, v)

    return best[0]

  def getAction(self, gameState):
    return self.a_minMax(gameState) #minMax(self, gameState, self.treeDepth)
    """
      Returns the minimax action from the current gameState using self.treeDepth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """


class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def a_minVal(self, gameState, ghostNo, depth, alpha, beta):
    #print "called min val at depth ", depth

    #if this is the last ghost, call max for pacman 
    if ghostNo == gameState.getNumAgents()-1:
      best_min1 = float('inf')
      for act in gameState.getLegalActions(ghostNo):
        nextState = gameState.generateSuccessor(ghostNo, act)
        v = self.a_maxVal(nextState, depth+1, alpha, beta)

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
        v = self.a_minVal(nextState, ghostNo+1, depth, alpha, beta)
       
        # v = min(v, min)
        if v < best_min2:
          best_min2 = v

        if best_min2 <= alpha: return best_min2
        beta = min(beta, best_min2)


      return best_min2


  def a_maxVal(self, gameState, depth, alpha, beta): #max is pacman agent

    if depth == self.treeDepth:
      return self.evaluationFunction(gameState)


    best_max = float('-inf')
    for act in gameState.getLegalActions(0):
      nextState = gameState.generateSuccessor(0, act)
      v = self.a_minVal(nextState, 1, depth, alpha, beta)

      #v = max(v, min)
      if v > best_max: 
        best_max = v

      if best_max >= beta: return best_max #if we already know it's a bad route, return
      alpha = max(alpha, best_max)

    return best_max


  def alpha_beta(self, gameState):
    best = (None, float('-inf'))
    alpha = float('-inf')
    beta = float('inf')
    for act in gameState.getLegalActions(0):
      nextState = gameState.generateSuccessor(0, act)
      v = self.a_minVal(nextState, 1, 0, alpha, beta)

      #v = max(minVal for actions)
      if v > best[1] and not act == 'Stop':
        best = (act, v)

    return best[0]


    
  def getAction(self, gameState):
    """
      Returns the minimax action using self.treeDepth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    return self.alpha_beta(gameState)

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def a_minVal(self, gameState, ghostNo, depth):
    #print "called min val at depth ", depth
    if gameState.isWin(): #if pacman has won already, just return evaluation
      return self.evaluationFunction(gameState)
    

    #if this is the last ghost, call max for pacman 
    if ghostNo == gameState.getNumAgents()-1:
      expect1 = 0
      nextActions = gameState.getLegalActions(ghostNo)
      for act in nextActions:
        nextState = gameState.generateSuccessor(ghostNo, act)
        v = self.a_maxVal(nextState, depth+1)
        expect1 += v

      if len(nextActions) == 0: nextActions.append('0') #if there's no actions, just append to avoid 1/0
      return expect1/len(nextActions) # average expectation

    #otherwise call for next ghost
    else:
      expect2 = 0
      nextActions = gameState.getLegalActions(ghostNo)
      for act in nextActions:
        nextState = gameState.generateSuccessor(ghostNo, act)
        v = self.a_minVal(nextState, ghostNo+1, depth)
        expect2 += v

      if len(nextActions) == 0: nextActions.append('0') 
      return expect2/len(nextActions)


  def a_maxVal(self, gameState, depth): #max is pacman agent

    if gameState.isWin(): #check if we've won already because if we have, just return
      return self.evaluationFunction(gameState)

    if depth == self.treeDepth: # if we've reached the max depth, treat like a terminal
      return self.evaluationFunction(gameState)


    # pacman makes best decision based on returns from expectation nodes
    best = float('-inf')
    for act in gameState.getLegalActions(0):
      nextState = gameState.generateSuccessor(0, act)
      v = self.a_minVal(nextState, 1, depth)
      if v > best: 
        best = v

    return best


  def expectimax(self, gameState): #top level of expectimax is pretty much the same as minimax and alpha-beta
    best = (None, float('-inf'))
    for act in gameState.getLegalActions(0):
      nextState = gameState.generateSuccessor(0, act)
      v = self.a_minVal(nextState, 1, 0)
      if v > best[1]:
        best = (act, v)

    return best[0]

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.treeDepth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    return self.expectimax(gameState)


# QUESTION 5

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
  

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

