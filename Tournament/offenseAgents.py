# Offense Agents.py

from reflexAgent import ReflexCaptureAgent
import random, time, util
from game import Directions
import game
from util import *
import datetime

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
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
      if a.isPacman and a.getPosition() != None:
        self.offOpps[i] += 1


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
            'offGhostState':10}




  def final(self, gameState):
    self.observationHistory= []
    #self.recordResults(gameState)



  def recordResults(self,gameState):
    f = open('records.txt' , 'a')
    now = datetime.datetime.now()

    f.write( now.strftime("%m-%d %H:%M")
             +" ("
            +str(self.getScore(gameState))
            +") { ")
    for w in self.getWeights():
      f.write(w+":"+str(self.getWeights()[w])+", ")
    f.write("}\n")