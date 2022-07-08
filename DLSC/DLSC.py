import numpy as np

#initialization
fig = 20
figno = np.array(range(fig, fig+16, 1))
caselist = ['CDL.5']
SCcaselist = ['WL1tanh']
p = [0.5]
casenumber = np.size(caselist, 0)
eachtrailplot = 0
eachlambdaplot = 0
eachiterprint = 1
numtrail=1
DLmaxiter =200
SCmaxiter=50
SCinternal=1
Dinitial = 'Xcols'
Zinitial = 'pinvDX'

#parameters for CDLWL
lstart = 1
lamda = 200e-0
SCDLlamdaratio = 1/1
c0 = [0.1]
rnumber = np.size(c0)+1
maxc = 20
multic = 1
iterc0 = [0.002]
Ntnconverge = 1e-6
Ntnitermax = 200
minNtnstep = 0.001
rzero = 1e-3
convergetol = 1e-3
convergetolSc = 1e-3

# trail begin
for trail in range(0, numtrail):
    PSNRoutput = np.zeros((1,casenumber))
    