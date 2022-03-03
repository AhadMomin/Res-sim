# Project 3


In this project, you will solve a two-dimensional, two-phase flow (i.e. oil and water) reservoir simulation in a heterogenuous reservoir with multiple wells.  Essentially, all of the functionality needed to do this was already implemented Project 2.  We will use real data from the Nechelik reservoir that we have looked at several times throughout the semester.

For this project, you should implement the class below `Project3()` which inherits from `TwoPhaseFlow` (which inherits from `TwoDimReservoir` and `BuckleyLevertt`).  

You will need to implement some functionality to read the porosity and permeability information from a file as you did in [Project2]

Other than reading the data from a file, you may not need to write any additional code for your simulation to work.  However, it might be a good idea to write a few plotting routines to produce some plots like this one

![img](images/contour.png)