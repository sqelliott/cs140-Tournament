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

    # Counter to hold our QValues ; initialize them to zero
    # keys are (state,action) tuples
    self.QValues = util.Counter()
    


  def getQValue(self, state, action):
    """
      Returns Q(state,action)
      Should return 0.0 if we never seen
      a state or (state,action) tuple
    """
    """Description:
    [Enter a description of what you did here.]

    To get the getQValue, get the value associated
    with the (state,action) pair in the QValues counter

    """
    """ YOUR CODE HERE """

    return self.QValues[(state,action)]

    util.raiseNotDefined()
    """ END CODE """



  def getValue(self, state):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    """Description:
    [Enter a description of what you did here.]

    Get all the legal actions
    Check if there are no actions at this state
    otherwise, we call getQValue of the current state
    with the optimal action, which comes from our 
    getPolicy

    """
    """ YOUR CODE HERE """

    actions = self.getLegalActions(state)
   
    if len(actions) == 0: return 0.0

    return self.getQValue(state, self.getPolicy(state))
    

    util.raiseNotDefined()
    """ END CODE """

  def getPolicy(self, state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    """Description:
    [Enter a description of what you did here.]

    Get all the actions of state ; check if there are any actions to take
    Create a list of (value,action) tuples for each action in legal actions
    get the maximum value and return the associated action

    """
    """ YOUR CODE HERE """

    actions = self.getLegalActions(state)

    if len(actions) == 0: return None

    actionValues = [ (self.getQValue(state,action),action) for action in actions]

    bestAction = max(actionValues)
    return bestAction[1]
    

    util.raiseNotDefined()
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
    [Enter a description of what you did here.]

    Do all the action checking shit
    if our coin flip is true, we choose a random action
    othewise, return the optimal policy

    """
    """ YOUR CODE HERE """

    # check for no legal actions
    if len(legalActions) == 0: return None

    # random action for no policy
    action = self.getPolicy(state)

    # return random action that 
    if util.flipCoin(self.epsilon):
       return random.choice(legalActions) 
    
    # return policy
    return action
   

    util.raiseNotDefined()
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
    [Enter a description of what you did here.]

    Update the QValue of our current state by getting the policy of the 
    nextstate, and adding the reward and the discounted QValue of (nextState,action)
    to our current QValue, using alpha normalize our QValue i.e. weight how much we consider
    old QValues versus new QValues

    """
    """ YOUR CODE HERE """

    next_action = self.getPolicy(nextState)

    self.QValues[(state,action)] = (1 - self.alpha) * self.QValues[(state, action)] + self.alpha * ( reward + self.discountRate * self.QValues[(nextState, next_action)] )
    return self.QValues[(state,action)]
    
    ### util.raiseNotDefined()
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
    self.weights = util.Counter()

  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    """Description:
    [Enter a description of what you did here.]

    Using the Counter __mult__ method, just multiply the two counters (weights and features)
    to perform the dot product operation and get the QValues
    """
    """ YOUR CODE HERE """

    return  self.weights * (self.featExtractor.getFeatures(state,action)) 
    util.raiseNotDefined()
    """ END CODE """

  def update(self, state, action, nextState, reward):
    """
       Should update your weights based on transition
    """
    """Description:
    [Enter a description of what you did here.]

    For each feature, calculate the correction value
    the weight for the current feature is the sum of itself and
    alpha * correction * the feature

    """
    """ YOUR CODE HERE """

    features = self.featExtractor.getFeatures(state,action)
    for feature in features:
       correction = reward + self.discountRate * self.getValue(nextState) - self.getQValue(state,action)
       self.weights[feature] += self.alpha * correction * features[feature]

    #util.raiseNotDefined()
    """ END CODE """

  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    PacmanQAgent.final(self, state)

    # did we finish training?
    if self.episodesSoFar == self.numTraining:
      # you might want to print your weights here for debugging
      print self.weights
      util.raiseNotDefined()
