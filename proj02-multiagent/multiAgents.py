# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

from functools import partial
from math import inf, log
import numpy as np

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        distToPacman = partial(manhattanDistance, newPos)
        
        def ghostF(ghost):
            dist = distToPacman(ghost.getPosition())
            if ghost.scaredTimer > dist:
                return inf
            if dist <= 1:
                return -inf
            return 0
        ghostScore = min(map(ghostF, newGhostStates))
        
        distToClosestFood = min(map(distToPacman, newFood.asList()), default=inf)
        closestFoodFeature = 1.0 / (1.0 + distToClosestFood)
        return successorGameState.getScore() + closestFoodFeature + ghostScore

def scoreEvaluationFunction(currentGameState: GameState):
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
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        ghostIdx = [i for i in range(1, gameState.getNumAgents())]

        def term(state, depth):
            return state.isWin() or state.isLose() or depth == self.depth

        def minValue(state, depth, ghost):
            if term(state, depth):
                return self.evaluationFunction(state)

            value = 1e9
            for action in state.getLegalActions(ghost):
                if ghost == ghostIdx[-1]:
                    value = min(value, maxValue(state.generateSuccessor(ghost, action), depth + 1))
                else:
                    value = min(value, minValue(state.generateSuccessor(ghost, action), depth, ghost + 1))

            return value

        def maxValue(state, depth):
            if term(state, depth):
                return self.evaluationFunction(state)

            value = -1e9
            for action in state.getLegalActions(0):
                value = max(value, minValue(state.generateSuccessor(0, action), depth, ghostIdx[0]))

            return value

        res = [(action, minValue(gameState.generateSuccessor(0, action), 0, ghostIdx[0])) \
            for action in gameState.getLegalActions(0)]
        res.sort(key=lambda k: k[1])

        return res[-1][0]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        curValue, alpha, beta = -1e9, -1e9, 1e9
        nextPacmanAction = Directions.STOP
        legalActions = gameState.getLegalActions(0).copy()

        for nextAction in legalActions:
            nextState = gameState.generateSuccessor(0, nextAction)
            nextValue = self.getNodeValue(nextState, 0, 1, alpha, beta)

            if nextValue > curValue:
                curValue, nextPacmanAction = nextValue, nextAction

            alpha = max(alpha, curValue)
        return nextPacmanAction

    def getNodeValue(self, gameState, depth = 0, agentIdx = 0, alpha = -1e9, beta = 1e9):
        """
        Using self-defined function, alphaValue(), betaValue() to choose the most appropriate action
        Only when it's the final state, can we get the value of each node, using the self.evaluationFunction(gameState)
        Otherwise we just get the alpha/beta value we defined here.
        """
        maxParty = [0, ]
        minParty = list(range(1, gameState.getNumAgents()))

        if depth == self.depth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        elif agentIdx in maxParty:
            return self.alphaValue(gameState, depth, agentIdx, alpha, beta)
        elif agentIdx in minParty:
            return self.betaValue(gameState, depth, agentIdx, alpha, beta)

    def alphaValue(self, gameState, depth, agentIdx, alpha = -1e9, beta = 1e9):
        """
        maxParty, search for maximums
        """
        value = -1e9
        legalActions = gameState.getLegalActions(agentIdx)
        for index, action in enumerate(legalActions):
            nextValue = self.getNodeValue(gameState.generateSuccessor(agentIdx, action), \
                depth, agentIdx + 1, alpha, beta)
            value = max(value, nextValue)
            if value > beta:  # next_agent in which party
                return value
            alpha = max(alpha, value)
        return value

    def betaValue(self, gameState, depth, agentIdx, alpha = -1e9, beta = 1e9):
        """
        minParty, search for minimums
        """
        value = 1e9
        legalActions = gameState.getLegalActions(agentIdx)
        for index, action in enumerate(legalActions):
            if agentIdx == gameState.getNumAgents() - 1:
                nextValue = self.getNodeValue(gameState.generateSuccessor(agentIdx, action), \
                    depth + 1, 0, alpha, beta)
                value = min(value, nextValue)  # begin next depth
                if value < alpha:
                    return value
            else:
                nextValue = self.getNodeValue(gameState.generateSuccessor(agentIdx, action), \
                    depth, agentIdx + 1, alpha, beta)
                value = min(value, nextValue)  # begin next depth
                if value < alpha:  # next agent goes on at the same depth
                    return value
            beta = min(beta, value)
        return value

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        maxValue = -1e9
        maxAction = Directions.STOP

        for action in gameState.getLegalActions(agentIndex=0):
            sucState = gameState.generateSuccessor(action=action, agentIndex=0)
            sucValue = self.expNode(sucState, curDepth=0, agentIndex=1)
            if sucValue > maxValue:
                maxValue = sucValue
                maxAction = action

        return maxAction

    def maxNode(self, gameState, curDepth):
        if curDepth == self.depth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        maxValue = -1e9
        for action in gameState.getLegalActions(agentIndex=0):
            sucState = gameState.generateSuccessor(action=action, agentIndex=0)
            sucValue = self.expNode(sucState, curDepth=curDepth, agentIndex=1)
            if sucValue > maxValue:
                maxValue = sucValue
        return maxValue

    def expNode(self, gameState, curDepth, agentIndex):
        if curDepth == self.depth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        numAction = len(gameState.getLegalActions(agentIndex=agentIndex))
        totalValue = 0.0
        numAgent = gameState.getNumAgents()
        for action in gameState.getLegalActions(agentIndex=agentIndex):
            sucState = gameState.generateSuccessor(agentIndex=agentIndex, action=action)
            if agentIndex == numAgent - 1:
                sucValue = self.maxNode(sucState, curDepth=curDepth + 1)
            else:
                sucValue = self.expNode(sucState, curDepth=curDepth, agentIndex=agentIndex + 1)
            totalValue += sucValue

        return totalValue / numAction


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()

    # Consts
    INF = 100000000.0  # Infinite value
    WEIGHT_FOOD = 10.0  # Food base value
    WEIGHT_GHOST = -10.0  # Ghost base value
    WEIGHT_SCARED_GHOST = 100.0  # Scared ghost base value

    # Base on gameState.getScore()
    score = currentGameState.getScore()

    # Evaluate the distance to the closest food
    distancesToFoodList = [util.manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]
    if len(distancesToFoodList) > 0:
        score += WEIGHT_FOOD / min(distancesToFoodList)
    else:
        score += WEIGHT_FOOD

    # Evaluate the distance to ghosts
    for ghost in newGhostStates:
        distance = manhattanDistance(newPos, ghost.getPosition())
        if distance > 0:
            if ghost.scaredTimer > 0:  # If scared, add points
                score += WEIGHT_SCARED_GHOST / distance
            else:  # If not, decrease points
                score += WEIGHT_GHOST / distance
        else:
            return -INF  # Pacman is dead at this point

    return score

# Abbreviation
better = betterEvaluationFunction
