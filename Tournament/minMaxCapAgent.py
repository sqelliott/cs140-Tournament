from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

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


class AlphaBetaAgent(MultiAgentSearchAgent):
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


    
  def getAction(self, gameState, maxDepth):
    """
      Returns the minimax action using self.treeDepth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    return self.alpha_beta(gameState, maxDepth)






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






