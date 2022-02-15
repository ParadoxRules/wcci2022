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

# Flips the kth bit of a genome (creates a new genome)
def flip(x, k):
  y = x.copy()
  y[k] = 1 if y[k]==0 else 0
  return y

# Generates the complement bitstring (creates a new genome)
def complement(x): return [1-v for v in x]

# For all logic operator tested on an array of boolean values
def for_all(a):
  for v in a:
    if(not v):
      return False
  return True

############### MUTATION FUNCTIONS ################

# Single bit mutation (creates a copy with a bit randomly flipped)
def single_bit_mutation(x):
  return flip(x, rand.randint(0,len(x)-1))


####################### Global variables ########################
TRACE = True # Tracing f value evolution
PRINT = True # Printing intermediate information

generated = [] # f value of each generated candidate solution at each iteration
best = [] # f value of the best generated candidate solution at each iteration
iter = 0 # Expended function evaluations

# Initializes global variables
def init():
  global generated, best, iter
  best = []
  generated = []
  iter = 0

# Evaluates the function on a given genome and stores information in the 
# global variables for further analysis if required
def evaluate(f, x):
  global generated, best, iter, TRACE
  fx = f(x)
  if( TRACE ): 
    generated.append(fx)
    if(iter==0 or fx>best[iter-1]):
      best.append(fx)
    else:
      best.append(best[iter-1])  
  iter += 1  
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
  return x, fx

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
# x: Initial point, a random point. It is required for defining the dimension of the space 
def GS1( f, evals, x ):
  global iter
  D = len(x) # Space dimension
  fx = evaluate(f, x)
  S1 = [x]
  fS1 = [fx]
  for i in range(evals-2):
    x = bitstring(D)
    fx = evaluate(f,x)
    S1.append(x)
    fS1.append(fx)
  # Computes schemata information
  M = len(S1)
  C = [[0 for k in range(D)], [0 for k in range(D)]]
  fH = [[0 for k in range(D)], [0 for k in range(D)]]
  for k in range(D):
    for i in range(M):
      fH[S1[i][k]][k] += fS1[i]
      C[S1[i][k]][k] += 1
  # Generates a candidate solution with the best genes
  y = []
  for k in range(D):
    y.append( 1 if(fH[1][k]/C[1][k] > fH[0][k]/C[0][k]) else 0 ) 
  return y, evaluate(f,y), S1, fS1

# The Global Search Algorithm with complement propossed by G. Venturini 1995
# Applies GS1 and compares the obtained solution with its complement and the 
# best candidate solution of the S1 set (same as the best solution found)
# f: Function to be optimized
# evals: Maximum number of fitness evaluations
# x: Initial point, a random point. It is required for defining the dimension of the space 
def GSC1( f, evals, x ):
  x, fx, S1, fS1 = GS1(f, evals-1, x)
  xc = complement(x)
  fxc = evaluate(f,xc)
  x, xc, fx, fxc = pick(x, xc, fx, fxc)
  for i in range(len(S1)):
    x, y, fx, fy = pick(x, S1[i], fx, fS1[i])
  return x, fx

############ GABO: Gene Analysis Bitstring Optimization #############
# Gene information
contribution = []
intron = []
separable = []

# initializes global variables
def init_gene(D):
  global contribution, intron, separable
  contribution = [[] for i in range(D)]
  intron = [True for i in range(D)]
  separable = [True for i in range(D)]

# Computes contribution information (relative to a value 1), i.e., some change 
# in the f value
# x: A candidate solution
# y: The candidate solution with the k-th bit flipped 
# fx: f value of x
# fy: f value of y
# k: Gene's locus
def C(x, y, fx, fy, k):
  global intron, contribution
  c = fx-fy
  intron[k] = intron[k] and c==0
  if(x[k]==0): c = -c 
  contribution[k].append(c)
  return c

# Gene characterization algorithm
# genome: An array with each gene information, see Gene class
# f: Function to be optimized
# x: initial point
# evals: Maximum number of fitness evaluations
def GCA( f, evals, x ):
  global iter, PRINT, separable
  D = len(x) # Space dimension
  fx = evaluate(f, x) if iter<evals else None
  if( iter<evals ):
    # The complement candidate solution (used for determining locus separability)
    xc = complement(x)
    fxc = evaluate(f,xc)
    x, xc, fx, fxc = pick(x, xc, fx, fxc)

  # Considers locus by locus (shuffles loci for reducing order effect)
  perm = permutation(D)
  a=0
  while(iter+2<=evals and a<D):
    k = perm[a]
    y = flip(x,k)
    fy = evaluate(f,y)
    yc = complement(y)
    fyc = evaluate(f,yc)    
    cx = C(x, y, fx, fy, k)
    cxc = C(xc, yc, fxc, fyc, k)
    separable[k] = separable[k] and (cx==cxc)
    w = x
    y, yc, fy, fyc = pick( y, yc, fy, fyc )
    x, y, fx, fy = pick( x, y, fx, fy )
    if(w!=x): xc, fxc = yc, fyc
    a += 1
  if(PRINT): print('Best GCA trial:', fx, x)
  return x, fx

# Checks the gene's contribution information to determine if the best bit value (allele)
# for the gene is 0: c<0, or 1: c>0. If c=0 then a gene behaves like an intron (neutral) so 
# bit is set to a random value. Returns a non-negative contribution (c) accordingly
def allele(c):
  if(c>0): return 1,c
  if(c<0): return 0,-c
  bit = 0 if randbool() else 1
  return bit, c

# Checks all the information about the gene's contribution to get the higher one,
# determines the best allele (0, 1, or None) for the gene  and the first time (in checking trials) it was reached
def best_allele(k):
  global contribution
  alle, cont = allele(contribution[k][0])
  trial = 0
  for i in range(1,len(contribution[k])):
    a, c = allele(contribution[k][i])
    if(c > cont): alle, cont, trial = a, c, i
  return alle, cont, trial

# Creates a candidate solution using locus contributions.
# For each locust gets the value of the bit (0, or 1) with the higher contribution
# that is stored in the delta list
# genome: An array with each gene information, see Gene class
# returns the best candidate solution so far and its f value
def success_trial():
  global contribution
  D = len(contribution)
  success = False
  k = 0
  while( k<D and not success ): 
    allele, cont, trial = best_allele(k)
    if(cont != 0): success = (trial+2 >= len(contribution[k]))
    k += 1
  return success


# Gene characterization analysis trials
# genome: An array with each gene information, see Gene class
# f: Function to be optimized
# x: initial point
# evals: Maximum number of fitness evaluations
def GCSA( f, evals, x ):
  global iter, PRINT, separable, intron, contribution
  D = len(x) # Space dimension

  x, fx = GCA(f, evals, x) # Best solution obtained with GCA

  if(for_all(separable) or for_all(intron)): return x, fx

  failTrials = 0 if success_trial() else 1
  while(iter+2<=evals and failTrials<3 ):
    y, fy = GCA(f, evals, bitstring(D)) # Best solution obtained with SLA
    failTrials = 0 if success_trial() else failTrials + 1
    x, y, fx, fy = pick(x,y,fx,fy)
  if(iter<evals):
    y = [best_allele(k)[0] for k in range(D)] 
    fy = evaluate(f,y)
    x, y, fx, fy = pick(x, y, fx, fy)
  if(PRINT): print('Best GCA trials:', fx, x)
  return x, fx


# Introns only search algorithm
# genome: An array with each gene information, see Gene class
# f: Function to be optimized
# x: initial point
# evals: Maximum number of fitness evaluations
# fx: f value in the x point, if available
def IOSA( f, evals, x, fx=None ):
  global iter, PRINT, intron
  if(not fx): fx = evaluate(f, x) # Evaluates f on x if not done
  D = len(x) # Space dimension

  introns = []
  for k in range(D):
    if(intron[k]): introns.append(k)

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
    if( C(x, y, fx, fy, k) != 0 ):
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
  init_gene(D) 

  x, fx = GCSA(f, evals, x) # Best solution obtained with SLA
  if(PRINT): print('******Best GCSA:', iter, fx, x)

  x, fx = IOSA(f, evals, x, fx) # Best solution for 'introns'
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

  # plt.savefig(PATH+function+'_'+str(DIM)+'_'+str(EVALS)+'.eps') 
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
PATH = "drive/My Drive/wcci2022/"
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

experiment1()
#binary_gabo_test('Royal Road 8', 480, 60000, 100)