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
    self.values = util.Counter() # create a value table to store known values

    """Description:
    
    When the agent is initialized, we want to find all the values of the states. This is done by iterating
    through all of the states and updating their values by choosing the best Q value corresponding to that state
    (unless it is a terminal state. In that case the value never changes). Two "Counter" dictionaries are used 
    -- one for the existing Q values and one for the updated ones. At the end of the iteration, the old values 
    are replaced by the new ones. The number of iterations is given as an input parameter. 

    """
    """ BEGIN CODING """

    # Want to value iterate for 'iters' number of times

    for skip in range(0, iters):

      newVals = util.Counter() # initialize a new counter for new values

      # iterate over all states 
      for state in self.mdp.getStates():

        qValList = []

        if self.mdp.isTerminal(state):

          newVals[state] = self.values[state]
          continue

        # get list of possible actions and iterate to find best q value
        possibleActions = self.mdp.getPossibleActions(state)
        for action in possibleActions:
          qValList.append(self.getQValue(state, action))

        newVals[state] = max(qValList)

      # Once we've iterated over all states, we've done one full iteration 
      # and now newVals contains V_i+1
      # so we now replace the old values with the next ones in the iteration
      self.values = newVals



    
    """ END CODE """

  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]

    """Description:

      doesn't need to be changed. Just return the value already stored.
    """

  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    """Description:

      The Q value is the expected value of taking an action. So, just iterate over 
      possible actions and multiply the value of the possible states with their probabilities. 
      As in Bellmans equations, we add possible future values multiplied by a discount rate as well.

    """
    """ YOUR CODE HERE """

    # What we want is the expected value of trying to do action 'action' from current state 'state'

    nextPossibleStatePairs = self.mdp.getTransitionStatesAndProbs(state, action)

    sumQ = 0

    # Iterate through possible actions and return the average value of them
    for (nextState, prob) in nextPossibleStatePairs:

      d = self.discountRate * self.values[nextState]
      sumQ += prob * (self.mdp.getReward(state, action, nextState) + d)


    return sumQ

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

      The policy at a state is just the direction corresponding to the best Q value for that state. 
      All we need to do is iterate over the Q values of the state, and choose the action accociated 
      with that Q value. 
      
    """
    """ YOUR CODE HERE """
    
    nextActs = self.mdp.getPossibleActions(state)

    if len(nextActs) == 0: return None

    qList = []
    for act in nextActs:
      t = (self.getQValue(state, act), act)
      qList.append(t)

    return max(qList)[1]

    """ END CODE """

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
