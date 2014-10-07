
# coding: utf-8

# # Some setup

# In[1]:

get_ipython().system(u'pip install --upgrade simworker')
get_ipython().system(u'pip install celery redis six')
import tellurium as te
import numpy as np
import matplotlib.pyplot as plt
import time
get_ipython().magic(u'matplotlib inline')
get_ipython().magic(u"config InlineBackend.figure_format='retina'")


# # Now let's define a model we will be using in the following examples

# In[2]:

model = '''
  #Reactions:
  J0:    -> S1;  J0_v0
  J1: S1 ->   ;  J1_k3*S1
  J2: S1 -> S2; (J2_k1*S1)*(1 + J2_c*S2^J2_q)
  J3: S2 ->   ;  J3_k2*S2

  # Species initializations:
  S1 = 0
  S2 = 1

  # Variable initializations:
  J0_v0 = 8
  J1_k3 = 1
  J2_k1 = 1
  J2_c  = 1
  J2_q  = 3
  J3_k2 = 5
'''

rr = te.loadAntimonyModel(model)
print rr.model.getGlobalParameterValues()

TIME_START = 0
TIME_END = 5
TIME_STEPS = 80

data = rr.simulate(TIME_START, TIME_END, TIME_STEPS)
c = np.copy(data) # need to copy results or else it gets changed later
te.plotArray(data)
sbml = rr.getSBML()


# # Now from our model let's create some noisy "experimental" data

# In[3]:

def makeNoise(results, mean=0, std=1):
    import numpy as np
    dim = results.shape
    numSpecies = dim[1] - 1
    numPoints = dim[0]
    noise = np.hstack([np.array([np.random.normal(mean, std, numPoints)]).T for s in range(numSpecies)])
    return noise

noise = makeNoise(c, std=0.2)
noisyResults = np.hstack((np.array([c[:, 0]]).T, c[:, 1:] + noise))
te.plotArray(noisyResults)


# # Now let's use an evolutionary algorithm to fit parameters
# 
# ## First, a helper simulation method and a fitness function

# In[4]:

def getSim(rr, params):
    rr.reset()
    rr.model.setGlobalParameterValues(params)
    return rr.simulate(TIME_START, TIME_END, TIME_STEPS)

def getFitness(obs):
    expected = noisyResults

    sumofsquares = 0
    for index, value in np.ndenumerate(obs):
            expectedValue = expected[index[0], index[1]]
            residual = (value - expectedValue)
            sumofsquares += residual ** 2

    return sumofsquares


# In[5]:

MAX_GENS = 50
POPULATION = 50


# In[6]:

import diffevolution
de = diffevolution.DiffEvolution(rr,
                   fitnessFcn=getFitness,
                   simFcn=getSim,
                   MAX_GENS=MAX_GENS,
                   POPULATION=POPULATION,
                   SAVE_RESULTS=True,
                   paramRangeDict={
                     'J0_v0': (8, 8), # Constrain a few variables to make this faster
                     'J1_k3': (1, 1),
                     'J2_k1': (1, 1),
                     #'J2_c': (1, 1),
                     #'J2_q': (3, 3),
                     #'J3_k2': (5, 5)
                   })
de.start()
de.plotFitnesses()
de.plotBest(observed=noisyResults)


# # Plotting correlations between fitted parameters

# In[7]:

def plotCorrelations(
    pInds, members,
    labels=None
):
    import matplotlib.pyplot as mplot
    import numpy as np

    f, axarr = mplot.subplots(
        len(pInds),
        len(pInds))

    for i, p1 in enumerate(pInds):
        for j, p2 in enumerate(pInds):
            k1 = np.array([m.params[p1] for m in members])
            k2 = np.array([m.params[p2] for m in members])

            axarr[i, j].scatter(
                k1,
                k2)
            
            if labels is not None:
                p1_label = labels[p1]
                p2_label = labels[p2]
            else:
                p1_label = str(p1)
                p2_label = str(p2)

            if (i == len(pInds) - 1):
                axarr[i, j].set_xlabel('%s' % (p2_label))
            if (j is 0):
                axarr[i, j].set_ylabel('%s' % (p1_label))



# In[8]:

parameter_names = ['J0_v0', 'J1_k3', 'J2_k1', 'J2_q', 'J3_k2', 'J2_c']

fitnesses = []
members = []
for g in de.generations:
    for f, m in zip(g['fitness'], g['members']):
        if f < 40:
            fitnesses.append(f)
            members.append(m)

pInds = [3, 4, 5]

plotCorrelations(pInds, members, labels=parameter_names)


# # Now let's run a distributed parameter fitting algorithm
# 
# ## We will need to define slightly different simuation and fitness function helpers

# In[9]:

expected = noisyResults
def asyncSim(rr, params):
    import tasks
    
    return tasks.rrChain.delay([
        ['load', sbml],
        ['reset'],
        [['model', 'setGlobalParameterValues'], params],
        ['simulate', TIME_START, TIME_END, TIME_STEPS]
    ])

def getAsyncFitness(obs):
    sumofsquares = 0
    for index, value in np.ndenumerate(obs):
        expectedValue = expected[index[0], index[1]]
        residual = (value - expectedValue)
        sumofsquares += residual ** 2
    return sumofsquares


# In[10]:

import diffevolution
MAX_GENS = 50
POPULATION = 50
deAsync = diffevolution.DiffEvolution(rr,
                   fitnessFcn=getAsyncFitness,
                   simFcn=asyncSim,
                   ASYNC=True,
                   SAVE_RESULTS=True,
                   MAX_GENS=MAX_GENS,
                   POPULATION=POPULATION,
                   paramRangeDict={
                     'J0_v0': (8, 8),
                     'J1_k3': (1, 1),
                     'J2_k1': (1, 1),
                     #'J2_c': (1, 1),
                     #'J2_q': (3, 3),
                     #'J3_k2': (5, 5)
                   })
deAsync.start()
deAsync.plotFitnesses()
deAsync.plotBest(observed=noisyResults)


# In[11]:

parameter_names = ['J0_v0', 'J1_k3', 'J2_k1', 'J2_q', 'J3_k2', 'J2_c']

fitnesses = []
members = []
for g in deAsync.generations:
    for f, m in zip(g['fitness'], g['members']):
        if f < 40:
            fitnesses.append(f)
            members.append(m)

pInds = [3, 4, 5]

plotCorrelations(pInds, members, labels=parameter_names)


# # Notice that the distributed parameter fit took longer? How could this happen? Let's investigate
# 
# ### First let's time how long it takes to do a simple add calculation

# In[12]:

import tasks
import time
start = time.time()
tasks.add.delay(2, 2).get()
end = time.time()
print '%s seconds' % str(end-start)


# There is some overhead in sending jobs and reading results. In our set up, there is also 0.5s rate limit on polling results.
# 
# As you do more jobs, the amount of overhead per job lessens

# In[13]:

import time
start = time.time()
numTrials = 30
asyncQueue = []
for i in range(numTrials):
    asyncQueue.append(tasks.add.delay(2, 2))
for job in asyncQueue:
    job.get()
end = time.time()
print '%s seconds' % str(end-start)
print '%s seconds per job' % str((end-start)/ numTrials) 


# In[14]:

# A slightly different method of calling many jobs, this is less performant for some reason
from celery import group
import time
start = time.time()
numTrials = 30
asyncQueue = []
for i in range(numTrials):
    asyncQueue.append(tasks.add.s(2, 2))

result = group(asyncQueue).apply_async()
print result.get()
end = time.time()
print '%s seconds' % str(end-start)
print '%s seconds per job' % str((end-start)/ numTrials) 


# # Now let's compare some simulations

# In[15]:

TIME_END = 100
NUM_SIMS = 300
import time
start = time.time()
asyncQueue = []
for i in range(NUM_SIMS):
    asyncQueue.append(tasks.rrChain.apply_async(
        ([['load', sbml],
          ['reset'],
          ['simulate', 0, TIME_END, 100]], ),
        serializer='json', compression='zlib'))

for job in asyncQueue:
    job.ready()
    job.get()
end = time.time()
print '%s seconds' % str(end-start)
print '%s seconds/sim' % str((end-start)/numSims)


# In[ ]:

import time
start = time.time()
for i in range(NUM_SIMS):
    #rr.load(sbml)
    rr.reset()
    rr.simulate(0, TIME_END, 100)
end = time.time()
print '%s seconds' % str(end-start)
print '%s seconds/sim' % str((end-start)/numSims)


# # For this case, it looks like the time it takes for a job to be sent and the results retrieved is longer than a single simulation
# 
# # What if each simulation took more time to complete?

# In[ ]:

TIME_END = 1000
NUM_SIMS = 300
import time
start = time.time()
asyncQueue = []
for i in range(NUM_SIMS):
    asyncQueue.append(tasks.rrChain.apply_async(
        ([['load', sbml],
          ['reset'],
          ['simulate', 0, TIME_END, 100]], ),
        serializer='json', compression='zlib'))

for job in asyncQueue:
    job.get()
end = time.time()
print '%s seconds' % str(end-start)
print '%s seconds/sim' % str((end-start)/numSims)


# In[ ]:

import time
start = time.time()
for i in range(NUM_SIMS):
    #rr.load(sbml)
    rr.reset()
    rr.simulate(0, TIME_END, 100)
end = time.time()
print '%s seconds' % str(end-start)
print '%s seconds/sim' % str((end-start)/numSims)

