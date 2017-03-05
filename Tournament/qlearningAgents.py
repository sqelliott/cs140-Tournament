# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *


import random,util,math

class QLearningAgent(ReinforcementAgent):
  """
    Q-Learning Agent

    Functions you should fill in:
      - getQValue
      - getAction
      - getValue
      - getPolicy
      - update

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.discountRate (discount rate)

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions
        for a state
  """
  def __init__(self, **args):
    "You can initialize Q-values here..."
    ReinforcementAgent.__init__(self, **args)

    # want to have a Q value table to keep track of Q values
    self.qValuesTable = util.Counter() 


  def getQValue(self, state, action):
    """
      Returns Q(state,action)
      Should return 0.0 if we never seen
      a state or (state,action) tuple
    """
    """Description:
    
      All we need to do is return the value we've already calculated for the Q value and 
      stored in the global Q value table. Make sure it's a float because that messes stuff
      up sometimes. 

    """
    """ YOUR CODE HERE """
    return float(self.qValuesTable[(state, action)])

    """ END CODE """



  def getValue(self, state):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    """Description:
    
      Getting the value of a state is the same as in value iteration. 
      Simply iterate through Q values of possible actions and choose the best one. 
    """
    """ YOUR CODE HERE """
    legalActions = self.getLegalActions(state)
    action = None

    if len(legalActions) == 0: return 0.0

    qList = []
    for act in legalActions:
      t = (self.getQValue(state, act), act)
      qList.append(t)

    return max(qList)[0]
    """ END CODE """

  def getPolicy(self, state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    """Description:

    Getting the policy is very similar to getting the value. Using an almost identical mechanism, 
    Iterate through all actions to find the best Q value and return the corresponding action. 

    """
    """ YOUR CODE HERE """
    legalActions = self.getLegalActions(state)
    action = None

    # if we're in a terminal state return None
    if len(legalActions) == 0: return action 

    qList = []
    for act in legalActions:
      t = (self.getQValue(state, act), act)
      qList.append(t)

    return max(qList)[1]


    """ END CODE """

  def getAction(self, state):
    """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.

      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
    """
    # Pick Action
    legalActions = self.getLegalActions(state)
    action = None

    """Description:
    
    Here we simply flip a coin to determine whether or not to act randomly, or to act optimally. 
    The result of the coin flip partially depends on the agent's epsilon. 

    """
    """ YOUR CODE HERE """
    if len(legalActions) == 0: return action

    if util.flipCoin(self.epsilon): 
      action = random.choice(legalActions)
    else:
      action = self.getPolicy(state)

    """ END CODE """

    return action

  def update(self, state, action, nextState, reward):
    """
      The parent class calls this to observe a
      state = action => nextState and reward transition.
      You should do your Q-Value update here

      NOTE: You should never call this function,
      it will be called on your behalf
    """
    """Description:
      
      The update function updates the Q value table. We do this by approximating future Q values and 
      adding the found reward, then adding to the known Q value. We do this with different weights (alpha)
      in order to be able to prefer new information to older information. 

      After the new Q value is found, replace it in the Q value table. 

          """
    """ YOUR CODE HERE """

    ql = []
    
    for a in self.getLegalActions(nextState): 
      ql.append(self.getQValue(nextState, a))

    if len(ql) == 0: ql.append(0.0)

    nextQ = max(ql)

    sample = reward + self.discountRate * nextQ
    oldQ = self.getQValue(state, action)
    newQ = (1 - self.alpha) * oldQ + self.alpha * sample

    self.qValuesTable[(state, action)] = float(newQ)

    """ END CODE """

class PacmanQAgent(QLearningAgent):
  "Exactly the same as QLearningAgent, but with different default parameters"

  def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
    """
    These default parameters can be changed from the pacman.py command line.
    For example, to change the exploration rate, try:
        python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    args['epsilon'] = epsilon
    args['gamma'] = gamma
    args['alpha'] = alpha
    args['numTraining'] = numTraining
    self.index = 0  # This is always Pacman
    QLearningAgent.__init__(self, **args)

  def getAction(self, state):
    """
    Simply calls the getAction method of QLearningAgent and then
    informs parent of action for Pacman.  Do not change or remove this
    method.
    """
    action = QLearningAgent.getAction(self,state)
    self.doAction(state,action)
    return action


class ApproximateQAgent(PacmanQAgent):
  """
     ApproximateQLearningAgent

     You should only have to overwrite getQValue
     and update.  All other QLearningAgent functions
     should work as is.
  """
  def __init__(self, extractor='IdentityExtractor', **args):
    self.featExtractor = util.lookup(extractor, globals())()
    PacmanQAgent.__init__(self, **args)

    # You might want to initialize weights here.
    self.featWeights = util.Counter()


  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    """Description:
      
      The Q value at the state is dot product of the feature weights and the feature values.
      Since * is overloaded, just return (w * featureVector)
    """
    """ YOUR CODE HERE """


    featureValues = self.featExtractor.getFeatures(state, action)
    return self.featWeights * featureValues


    """ END CODE """

  def update(self, state, action, nextState, reward):
    """
       Should update your weights based on transition
    """
    """Description:
    
      The idea here is that instead of learning Q values directly, we're learning weights that when 
      multiplied by feature vectors give Q values. 

      To learn weights, we correct known weights in a similar way to regular Q learning. 

    """
    """ YOUR CODE HERE """
    featDict = self.featExtractor.getFeatures(state, action)
    for feat in featDict.keys():
      correction = (reward + self.discountRate * self.getValue(nextState)) - self.getQValue(state, action)
      self.featWeights[feat] += self.alpha * correction * featDict[feat]

    """ END CODE """

  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    PacmanQAgent.final(self, state)

    # did we finish training?
    if self.episodesSoFar == self.numTraining:
      # you might want to print your weights here for debugging
      #util.raiseNotDefined()
      print 'print weights'
      print self.featWeights