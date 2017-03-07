  # valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp, discountRate = 0.9, iters = 100):
    """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.

      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
    self.mdp = mdp
    self.discountRate = discountRate
    self.iters = iters
    self.values = util.Counter() # A Counter is a dict with default 0

    """Description:
    
    Perform value iteration an abritrary amount of times.
    For each state, set its value to the maximum QValue based on all
    the states actions

    After each iteration of value iteration, update the values

    """
    """ YOUR CODE HERE """
    ## import pdb;pdb.set_trace()
    states = mdp.getStates()
    self.valuesNew = util.Counter() # maintain old values

    # value iteration 
    for i in range(0,iters):
      #update all the states
      for state in states:
        # terminal states stay the same
        if state == 'TERMINAL_STATE':
          self.valuesNew[state] = self.values[state]
        else:
          values = []
          # get qValues for all actions from state
          for action in mdp.getPossibleActions(state):
            values.append(self.getQValue(state,action))

          self.valuesNew[state] = max(values)
      # update values
      for state in states:
        self.values[state] = self.valuesNew[state]
      

    ####util.raiseNotDefined()
    """ END CODE """

  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]


  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    """Description:
    [Enter a description of what you did here.]

    We calculate the expected value of taking a action from a state,
    based on probabilities that taking an action at a state will take us
    to a different state
    """
    " YOUR CODE HERE "



    mdp = self.mdp
    gamma = self.discountRate

    next_states = mdp.getTransitionStatesAndProbs(state,action)
    qValue = 0

    for nextState, prob in next_states:
      reward = mdp.getReward(state,action, nextState)
      qValue += prob * ( reward + gamma * self.values[nextState])

    return qValue


    util.raiseNotDefined()
    """ END CODE """

  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """

    """Description:
    Get the possible actions from state
    If there are no actions, return None
    Loop through actions to see which as best Q-value
    """
    """ YOUR CODE HERE """
    import random

    actions = self.mdp.getPossibleActions(state)
    if len(actions) == 0:
      return None

    actionValues = []

    for action in actions:
      actionValues.append( (self.getQValue(state,action)))

    bestValue = max(actionValues)
    bestIndices = [index for index in range(len(actionValues)) if actionValues[index]==bestValue]
    index = random.choice(bestIndices)

    return actions[index]




    util.raiseNotDefined()
    """ END CODE """

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
