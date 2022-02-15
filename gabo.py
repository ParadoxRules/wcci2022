# Copyright (c)
# Authors: XXX  
# E-mails: XXXX
# All rights reserved.
# Licence
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer 
# in the documentation and/or other materials provided with the distribution.
# Neither the name of the copyright owners, their employers, nor the names of its contributors may be used to endorse or 
# promote products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, 
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT OWNERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import random as rand

from inspect import signature
import matplotlib.pyplot as plt

############### USEFUL FUNCTIONS ################
# Generates a boolean value according to probability p ( True with probability p, False otherwise )
def randbool(p=0.5):
  return (rand.random() < p)

# A permutation of n elements
def permutation(n):
  x = [i for i in range(0,n)]
  rand.shuffle(x)
  return x

############### MUTATION FUNCTIONS ################
# Flip a bit
def flip(x, k):
  y = x.copy()
  y[k] = 1 if y[k]==0 else 0
  return y

# Flip multiple bits
def multi_flip(x, indices):
  y = x.copy()
  for k in indices:
    y[k] = 1 if y[k]==0 else 0
  return y

# Single bit mutation
def single_bit_mutation(x):
  return flip(x, rand.randint(0,len(x)-1))

# Bit mutation
def bit_mutation(x):
  y = x.copy()
  D = len(x)
  p = 1/D
  c = 0
  while(c==0):
    for k in range(D):
      if(randbool(p)): 
        y[k] = 1 if y[k]==0 else 0
        c += 1
  return y

####################### Global variables ########################
TRACE = True # Tracing f value evolution
PRINT = True # Printing intermediate information

best = [] # f value of the best candidate solution at each iteration
best_x = None # Best candidate solution found
best_f = None # f value of the best candidate solution 
iter = 0 # Curren iteration (number of f evaluations)

# Initializes global variables
def init():
  global best, best_x, best_f, iter
  best = []
  best_x = None
  best_f = None
  iter = 0

# Evaluates the fitness function. Stores information in the generated and values 
# global variables for further analysis if required
def evaluate(f, x):
  global best_f, best_x, iter, TRACE
  fx = f(x)
  iter += 1  
  if(iter==1 or fx>=best_f): 
      best_x = x
      best_f = fx
  if( TRACE ): best.append(best_f) # Tracing evolution information or not
  return fx

#################### LITERATURE ALGORITHMS #####################
# Classical Hill Climbing Algorithm with neutral mutations
# Sorts two candidate solutions according to their f values
# If x is the current solution and y is a new candidate solution obtained by a 
# mutation operation, this method picks y if it has equal or higher f value 
# (neutral mutations are allowed) otherwise returns x (hill climbing replacement)
def pick( x, y, fx, fy ):
  if(fy>=fx): return y, x, fy, fx
  return x, y, fx, fy

# f: Function to be optimized
# mutation: Mutation operation
# evals: Maximum number of fitness evaluations
# x: Initial point
# fx: f value at point x
def HC( f, mutation, evals, x, fx=None):
  global iter
  if(not fx): fx = evaluate(f, x)
  while(iter<evals):
    y = mutation(x)
    fy = evaluate(f,y)
    x, y, fx, fy = pick( x, y, fx, fy )
  return x

# The HC algorithm suggested by Richard Palmer [15], that Forrest and
# Mitchell termed "random mutation hill-climbing" (RMHC)
# f: Function to be optimized
# evals: Maximum number of fitness evaluations
# x: Initial point
# fx: f value at point x
def RMHC( f, evals, x, fx=None): return HC(f, single_bit_mutation, evals, x, fx )

# The Global Search Algorithm for GA-easy functions propossed by Das and Whitley 1991
# Tries only order 1-schemas
# f: Function to be optimized
# evals: Maximum number of fitness evaluations
# x: Initial point
# fx: f value at point x
def GS1( f, evals, x, fx=None ):
  global iter
  D = len(x) # Space dimension
  if(not fx): fx = evaluate(f, x)
  S1 = [x]
  fS1 = [fx]
  for i in range(iter, evals-1):
    x = bitstring(D)
    fx = evaluate(f,x)
    S1.append(x)
    fS1.append(fx)
  
  M = len(S1)
  C = [[0 for k in range(D)], [0 for k in range(D)]]
  fH = [[0 for k in range(D)], [0 for k in range(D)]]
  y = []
  for k in range(D):
    for i in range(M):
      fH[S1[i][k]][k] += fS1[i]
      C[S1[i][k]][k] += 1
    y.append( 1 if(fH[1][k]/C[1][k] > fH[0][k]/C[0][k]) else 0 ) 
  return y, evaluate(f,y)

# The Global Search Algorithm with complement propossed by G. Venturini 1995
# Applies GS1 and compares the obtained solution with its complement and the 
# best candidate solution of the S1 set (same as the best solution found)
# f: Function to be optimized
# evals: Maximum number of fitness evaluations
# x: Initial point
# fx: f value at point x
def GSC1( f, evals, x, fx=None ):
  global best_x, best_f
  x, fx = GS1(f, evals-1, x, fx)
  xc = [1-x[k] for k in range(len(x))]
  fxc = evaluate(f,xc)
  return best_x, best_f

############ HILL CLIMBING ALGORITHM WITH LOCUS ANALYSIS #############
# Gene information
class Gene:
  # locus: Index of the gene (locus) 
  # delta: Contribution of the gene to the f value change when the gene takes the 1 value
  # separable: Indicates if the gene looks like separable form the rest of the genome
  # intron: Indicates if the gene has not produced any contribution to the f value (all contributions are zero)
  def __init__(self, locus):
    self.locus = locus
    self.delta = []
    self.separable = True
    self.intron = True
  
  # Computes contribution information (relative to a value 1), i.e., some change 
  # in the f value
  # x: A candidate solution
  # y: The candidate solution with the k-th bit flipped 
  # fx: f value of x
  # fy: f value of y
  def contribution(self, x, y, fx, fy):
    d = fx-fy
    self.intron = self.intron and d==0
    if(x[self.locus]==1): self.delta.append(d)
    else: self.delta.append(-d)
    return d

  # Computes gene contribution on a genome and its complement (consider k=3, starting at 0)
  # x: Current candidate solution (for example, x=10010110)
  # y: Current candidate solution with the gene flipped (for the example, y=10000110)
  # xc: Complement of the current candidate solution (for the example, xc=01101001)
  # yc: Complement of the current candidate solution with the gene flipped (for the example, yc=01111001)
  # fx: f value of x
  # fy: f value of y
  # fxc: f value of xc
  # fyc: f value of yc
  # Returns the candidate solutions according to f value and a flag indicating if there
  # was some change in the contribution of the locus 
  def update(self, x, y, xc, yc, fx, fy, fxc, fyc):
    d1 = self.contribution(x, y, fx, fy)
    d2 = self.contribution(yc, xc, fyc, fxc)
    self.separable = self.separable and d1==d2

    x, xc, fx, fxc = pick( x, xc, fx, fxc )
    w = x
    y, yc, fy, fyc = pick( y, yc, fy, fyc )
    x, y, fx, fy = pick( x, y, fx, fy )
    if(w!=x): xc, yc, fxc, fyc = yc, xc, fyc, fxc
    return x, y, xc, yc, fx, fy, fxc, fyc

  # Checks the gene's contribution information to determine if the best bit value (allele)
  # for the gene is 0: d<0, or 1: d>0. If d=0 then a gene behaves like an intron (neutral) so 
  # bit is set to None. Returns a non-negative contribution (d) accordingly
  def check(self, d):
    bit = None
    if(d>0):
      bit = 1
    elif(d<0):
      bit = 0
      d = -d
    return bit, d

  # Checks all the information about the gene's contribution to get the higher one,
  # determines the best allele (0, 1, or None) for the gene  and the first time (in checking trials) it was reached
  def best(self):
    b, d = self.check(self.delta[0])
    i = 0
    for k in range(1,len(self.delta)):
      bn, dn = self.check(self.delta[k])
      if(dn > d): b, d, i = bn, dn, k
    return b, d, i

  # Computes contribution information of flipping the gene, and all
  # the other genes and determines if there is some change in the contribution
  # value of the gene to the f value. If so there is some relation with some or
  # all the other genes, otherwise there is not relation or ot is hard to determine
  # (it is hard when flipping the gene produces a zero contribution in the f value)
  # f: Function being optimized
  # x: Current candidate solution
  # xc: The 'complement' of the current solution (all components flipped)
  # fx: f value of x
  # fxc: f value of xc
  # Returns the candidate solution and its complement according to f value 
  def analyze(self, f, x, xc, fx, fxc):
    D = len(x)
    y = flip(x,self.locus)
    fy = evaluate(f,y)
    yc = flip(xc,self.locus)
    fyc = evaluate(f,yc)    
    x, y, xc, yc, fx, fy, fxc, fyc = self.update(x, y, xc, yc, fx, fy, fxc, fyc)
    return x, xc, fx, fxc    

########## GABO: Gene Analysis Based Optimization Algorithm ###########
def fully_separable(genome):
  for gene in genome:
    if(not gene.separable):
      return False
  return True

def fully_introns(genome):
  for gene in genome:
    if(not gene.intron):
      return False
  return True

# Creates a candidate solution using locus contributions.
# For each locust gets the value of the bit (0, or 1) with the higher contribution
# that is stored in the delta list
# genome: An array with each gene information, see Gene class
# returns the best candidate solution so far and its f value
def succesful_trial(genome):
  D = len(genome)
  succesful = False # If additional SCA trial must carried on
  k = 0
  while( k<D and not succesful ): 
    b, d, i = genome[k].best()
    if(d != 0): succesful = (i+2 >= len(genome[k].delta))
    k += 1
  return succesful

# Gene characterization algorithm
# genome: An array with each gene information, see Gene class
# f: Function to be optimized
# x: initial point
# evals: Maximum number of fitness evaluations
# fx: f value in the x point
def GCA( genome, f, evals, x=None ):
  global iter
  D = len(genome) # Space dimension
  if( not x ): x = bitstring(D)
  fx = evaluate(f, x)
  if( iter<evals ):
    # The complement candidate solution (used for determining locus separability)
    xc = [1-x[k] for k in range(D)]
    fxc = evaluate(f,xc)
    x, xc, fx, fxc = pick(x, xc, fx, fxc)

  # Considers locus by locus (shuffles loci for reducing order effect)
  perm = permutation(D)
  a=0
  while(iter+2<=evals and a<D):
    k = perm[a]
    x, xc, fx, fxc = genome[k].analyze(f, x, xc, fx, fxc)
    a += 1

  return x, fx

# Gene characterization analysis trials
# genome: An array with each gene information, see Gene class
# f: Function to be optimized
# x: initial point
# evals: Maximum number of fitness evaluations
def GCSA( genome, f, evals, x ):
  global iter
  D = len(x) # Space dimension

  x, fx = GCA(genome, f, evals, x) # Best solution obtained with GCA

  if(fully_separable(genome) or fully_introns(genome)): return x, fx

  failTrials = 0 if new_trial(genome) else 1
  while(iter+2<=evals and failTrials<3 ):
    y, fy = GCA(genome, f, evals) # Best solution obtained with SLA
    failTrials = 0 if new_trial(genome) else faliTrial + 1
    x, y, fx, fy = pick(x,y,fx,fy)
  if(iter<evals):
    y = bitstring(D)
    for k in range(D): 
      b = genome[k].best()[0]
      if(b != None): y[k] = b
    fy = evaluate(f,x)
    x, y, fx, fy = pick(x, y, fx, fy)
  return x, fx


# Introns only search algorithm
# genome: An array with each gene information, see Gene class
# f: Function to be optimized
# x: initial point
# evals: Maximum number of fitness evaluations
# fx: f value in the x point, if available
def IOSA( genome, f, evals, x, fx=None ):
  global iter, PRINT
  if(not fx): fx = evaluate(f, x) # Evaluates f on x if not done
  D = len(x) # Space dimension

  introns = []
  for k in range(D):
    if(genome[k].intron): introns.append(k)

  if(PRINT): 
    print('Introns:', introns)
    print('Removes: (intron, at iteration)')
  
  N = len(introns)
  while(iter<evals and N>0):
    j = rand.randint(0,N-1)
    k = introns[j]
    y = flip(x,k)
    fy = evaluate(f,y)
    x, y, fx, fy = pick(x, y, fx, fy)
    # Checks if the locus is not neutral anymore and removes it
    if( genome[k].contribution(x, y, fx, fy) != 0 ):
      introns.pop(j)
      N-=1
      if(PRINT): print('(', k, ',', iter, ')', sep='')
  if(PRINT): print()
  return x, fx

# f: Function to be optimized
# x: initial point
# evals: Maximum number of fitness evaluations
def GABO( f, evals, x ):
  global iter, generated, PRINT

  D = len(x) # Space dimension

  # Initializes component information
  genome = [Gene(k) for k in range(D)] 

  x, fx = GCSA(genome, f, evals, x) # Best solution obtained with SLA
  if(PRINT): print('******Best GCA:', iter, fx, x)

  x, fx = IOSA(genome, f, evals, x, fx) # Best solution for 'introns'
  if(PRINT): print('******Best IOSA:', iter, fx, x) # Remove this line if printing is not required

  return x

##################### BINARY STRING SPACE #####################
# A bitstring of length DIM
def bitstring(DIM):
  return [1 if randbool() else 0 for i in range(0,DIM)]

# MaxOnes function
def maxones(x, start=0, end=-1):
  if(end<0): end = len(x)
  s = 0
  for i in range(start,end): s += x[i]
  return s

# Goldberg's 3-Deceptive function
def deceptive(x, start=0, end=-1):
  if(end<0): end = len(x)
  switcher = {
    0: 28,
    1: 26,
    2: 22,
    3: 0,
    4: 14,
    5: 0,
    6: 0,
    7: 30
  }
  size=3
  c = 0
  i = start
  while i<end:
    d=0
    b=1
    for k in range(i,i+size):
      d += b if x[k] else 0
      b *= 2
    c += switcher.get(d, 0)
    i += size  
  return c

# Goldberg's Boundedly-Deceptive function
def generic_boundedly(x, size, start=0, end=-1):
  if(end<0): end = len(x)
  c = 0
  i = start
  while i<end:
    u = 0
    for k in range(i,i+size):
      u += 1 if x[k] else 0
    c += u if u==size else size-1-u
    i += size
  return c

def boundedly(x, start=0, end=-1): return generic_boundedly(x,4,start,end)

# A combination of all the previous bit functions, each block of 20 bits is defined as follow
# 0..4 : maxones
# 5..7 : deceptive3
# 8..11 : boundedly
# 12..19 : royalroad8
def mixed(x, start=0, end=-1):
  if(end==-1): end = len(x)
  f = 0
  while(start<end):
    f += maxones(x,start,start+5) + deceptive(x,start+5,start+8) + boundedly(x,start+8,start+12) + royalroad8(x,start+12,start+20)
    start += 20
  return f 

# Forrest's Royal Road function
def generic_royalroad(x, size, start=0, end=-1):
  if(end<0): end = len(x)
  c = 0
  i = start
  while i<end:
    start += size
    while i<start and x[i]:
      i+=1
    c += size if i==start else 0
    i = start
  return c

def royalroad8(x,start=0,end=-1): return generic_royalroad(x,8,start,end)

def royalroad16(x,start=0,end=-1): return generic_royalroad(x,16,start,end)

##################### PLOTTING #####################
def plot( method, function, fx, DIM=0, EVALS=0, log=False ): 
  global PATH
  x = np.empty(len(fx))
  y = np.empty(len(fx))
  for i in range(len(x)):
    x[i] = i
    y[i] = fx[i]
   
  plt.plot(x, y, label=method)
  if( log ): plt.yscale('log')
  plt.xlabel( function + ' evaluations')  
  plt.ylabel( function + ' value' )  

  plt.show()

##################### MAIN #####################

# Testing a method on a binary function 
def method_test(method, function, x, EVALS):
  global best, PRINT

  init()
  DIM = len(x)
  algorithm = {'GSC1':GSC1, 'RMHC':RMHC, 'GABO':GABO}
  functions = {'Max Ones':maxones, 'Deceptive3':deceptive, 'Boundedly':boundedly, 'Royal Road 8':royalroad8, 'Royal Road 16':royalroad16, 'Mixed':mixed}

  alg = algorithm[method]
  f = functions[function]
  
  y = alg( f, EVALS, x )

  if(PRINT):
    print('*****Best found by',method,':') 
    print('Starts at:', best[0])
    print('Ends at:', best[len(best)-1])
    print('Candidate solution:', y)

  return y

def stats(results, opt):
  EXP = len(results)
  M = 0
  for r in range(EXP):
    if(M<len(results[r])): M = len(results[r]) 
  avg = [0 for i in range(M)]
  found_opt = []
  evals = []
  for r in range(EXP):
    flag = True
    Mr = len(results[r])
    evals.append(Mr)
    for i in range(Mr):
      if(opt==results[r][i] and flag): 
        found_opt.append(i+1)
        flag = False
      avg[i] += results[r][i]
    best = results[r][Mr-1]
    for i in range(Mr,M): avg[i] += best
  for i in range(M): avg[i] /= EXP
  std = [0 for i in range(M)]
  for r in range(EXP):
    Mr = len(results[r])
    for i in range(Mr): std[i] += (results[r][i]-avg[i])**2
    best = results[r][Mr-1]
    for i in range(Mr,M): std[i] += (best-avg[i])**2
  for i in range(M): std[i] = (std[i]/EXP)**0.5

  avg_iter = 0
  std_iter = 0
  for i in evals: avg_iter += i
  avg_iter /= EXP
  for i in evals: std_iter += (i-avg_iter)**2
  std_iter = (std_iter/EXP)**0.5

  avg_SR_iter = 0
  std_SR_iter = 0
  SR = len(found_opt) # Success Rate
  if( SR>0 ):
    for i in found_opt: avg_SR_iter += i
    avg_SR_iter /= SR
    for i in found_opt: std_SR_iter += (i-avg_SR_iter)**2
    std_SR_iter = (std_SR_iter/SR)**0.5
  
  return avg, std, avg_iter, std_iter, SR/EXP, avg_SR_iter, std_SR_iter, evals, found_opt

def print_stats(method, function, DIM, EVALS, stats):
  avg, std, avg_iter, std_iter, SR, avg_SR_iter, std_SR_iter, evals, found_opt = stats
  I = len(avg)-1
  print('Best found by', method, ':', avg[I], '+/-', std[I])
  print('Iters Expended:', evals)
  print('Iters Opt:', found_opt)
  print('Success Rate:', SR)
  print('Evaluations:', avg_iter, '+/-', std_iter)
  print('Optimum found at iteration:', avg_SR_iter, '+/-', std_SR_iter)
  plot(method, function, avg, DIM, EVALS, False)


# Testing on binary functions
# function : Name of the function to be optimized
# DIM : Space dimension
# EVALS : Maximum number of evaluations
# EXP : Number of experiments
def binary_cross_test(function, DIM, EVALS, EXP):
  global best, PRINT

  optima = {'Max Ones':DIM, 'Deceptive3':10*DIM, 'Boundedly':DIM, 'Royal Road 8':DIM, 'Royal Road 16':DIM, 'Mixed':47*DIM//20}
  opt = optima[function]

  hcla_results = [] # Results of the proposed approach HCLA
  gsc1_results = [] # Results of the GSC1 algorithm
  rmhc_results = [] # Results of the RMHC algorithm 

  for r in range(EXP):
    x = bitstring(DIM) # Same initial candidate solution for all algorithm
 
    y = method_test('GABO', function, x, EVALS )
    hcla_results.append(best)

    y = method_test('GSC1', function, x, EVALS )
    gsc1_results.append(best)

    y = method_test('RMHC', function, x, EVALS )
    rmhc_results.append(best)
    
  print_stats( 'GABO', function, DIM, EVALS, stats(hcla_results, opt) )
  print_stats( 'GSC1', function, DIM, EVALS, stats(gsc1_results, opt) )
  print_stats( 'RMHC', function, DIM, EVALS, stats(rmhc_results, opt) )

# Testing on binary functions
# function : Name of the function to be optimized
# DIM : Space dimension
# EVALS : Maximum number of evaluations
# EXP : Number of experiments
def binary_gabo_test(function, DIM, EVALS, EXP):
  global best, PRINT

  optima = {'Max Ones':DIM, 'Deceptive3':10*DIM, 'Boundedly':DIM, 'Royal Road 8':DIM, 'Royal Road 16':DIM, 'Mixed':47*DIM//20}
  opt = optima[function]

  hcla_results = [] # Results of the proposed approach HCLA
  gsc1_results = [] # Results of the GSC1 algorithm
  rmhc_results = [] # Results of the RMHC algorithm 

  for r in range(EXP):
    x = bitstring(DIM) # Initial candidate solution
 
    y = method_test('GABO', function, x, EVALS )
    hcla_results.append(best)
    
  print_stats( 'GABO', function, DIM, EVALS, stats(hcla_results, opt) )

#main
PRINT = False # Set to True if traced information must be printed
function = ['Max Ones', 'Deceptive3', 'Boundedly', 'Royal Road 8', 'Mixed'] # Function under optimization

def experiment1():
  global function
  DIM = [100, 30, 40, 64, 100] # Space dimension
  EVALS = [205,500,1000,6400,6400]
  EXP = 100 # Number of experiments
  for k in range(len(function)):
    print('//////////////////////',function[k],'//////////////////////')
    print('%%%%%%%%%%%%%%%',EVALS[k],'%%%%%%%%%%%%%%')
    binary_cross_test(function[k], DIM[k], EVALS[k], EXP)

def experiment2():
  global function
  DIM = [120, 240, 360, 480] # Space dimension
  EXP = 100 # Number of experiments
  for k in range(1,len(function)):
    print('//////////////////////',function[k],'//////////////////////')
    for D in DIM:
      print('//////////////////////',D,'//////////////////////')
      EVALS =  150*D
      print('%%%%%%%%%%%%%%%',EVALS,'%%%%%%%%%%%%%%')
      binary_gabo_test(function[k], D, EVALS, EXP)
