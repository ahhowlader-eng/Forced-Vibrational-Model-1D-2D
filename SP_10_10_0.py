import numpy as np
import time
from scipy.io import loadmat

# --------------------------------------------------
# Initialization
# --------------------------------------------------
t0 = time.time()

data = loadmat('n1010_per_1000_0_exact.mat')
M  = data['M']
LN = int(data['LN'][0][0])
mi = data['mi']
xx = data['xx']
yy = data['yy']

LX = 82
LY = 50

# --------------------------------------------------
# Spring constants
# --------------------------------------------------
Kin  = 245 * 0.994501659
Kra  = 365 * 0.994501659
Kout = 98.2 * 0.994501659

K2in  = -32.3 * 0.983412758
K2ra  =  88.0 * 0.983412758
K2out =  -4.0 * 0.983412758

K3in  = -52.5 * 0.977821151
K3ra  =  30.0 * 0.977821151
K3out =   1.5 * 0.977821151

K4in  =  22.9 * 0.96085111
K4ra  = -19.2 * 0.96085111
K4out =  -5.8 * 0.96085111

M1 = 1.993e-26
F0 = 1.0
TOTAL_ATOM = LN

# --------------------------------------------------
# Allocate spring matrices
# --------------------------------------------------
shape = (LY + 8, LX + 8)

def Z():
    return np.zeros(shape)

Kin1 = Z(); Kin2 = Z(); Kin21 = Z(); Kin22 = Z(); Kin23 = Z()
Kin31 = Z(); Kin32 = Z()
Kin411 = Z(); Kin412 = Z(); Kin413 = Z()
Kin421 = Z(); Kin422 = Z(); Kin423 = Z()
Kin431 = Z(); Kin432 = Z(); Kin433 = Z()
Kin441 = Z(); Kin442 = Z(); Kin443 = Z()

Kra1 = Z(); Kra2 = Z(); Kra21 = Z(); Kra22 = Z(); Kra23 = Z()
Kra31 = Z(); Kra32 = Z()
Kra411 = Z(); Kra412 = Z(); Kra413 = Z()
Kra421 = Z(); Kra422 = Z(); Kra423 = Z()
Kra431 = Z(); Kra432 = Z(); Kra433 = Z()
Kra441 = Z(); Kra442 = Z(); Kra443 = Z()

Kout1 = Z(); Kout2 = Z(); Kout21 = Z(); Kout22 = Z(); Kout23 = Z()
Kout31 = Z(); Kout32 = Z()
Kout411 = Z(); Kout412 = Z(); Kout413 = Z()
Kout421 = Z(); Kout422 = Z(); Kout423 = Z()
Kout431 = Z(); Kout432 = Z(); Kout433 = Z()
Kout441 = Z(); Kout442 = Z(); Kout443 = Z()

# --------------------------------------------------
# Helper for periodic copying
# --------------------------------------------------
def periodic_copy(A, yend):
    A[:yend, 27:30] = A[:yend, 57:60]
    A[:yend, 60:63] = A[:yend, 30:33]

# --------------------------------------------------
# Example spring block (Kin2 / Kra2 / Kout2)
# --------------------------------------------------
for X in range(30, 60):
    for Y in range(0, LY + 7):
        if M[Y, X] <= 0:
            continue
        if (M[Y+1, X] > 0 or M[Y+1, X-1] > 0 or M[Y+1, X+1] > 0):
            Kin2[Y, X]  = Kin
            Kra2[Y, X]  = Kra
            Kout2[Y, X] = Kout

periodic_copy(Kin2, LY+7)
periodic_copy(Kra2, LY+7)
periodic_copy(Kout2, LY+7)

# --------------------------------------------------
# Time & frequency parameters
# --------------------------------------------------
TAU = 0.01e-13
SQM = np.sqrt(M1)
TM  = TAU / M1
WS  = 1e12

P1 = 8
WE = 3.25e14
WSTEP = 5e12
DIV = 5e12

W = 2.5431e14
DN  = int(round(W / DIV)) + 1
ECST = int(2.0 * np.pi * DN / (W * TAU))

# --------------------------------------------------
# Mode loop
# --------------------------------------------------
for P in range(P1 + 1):

    U0X = Z(); U1X = Z(); VDX = Z(); VX = Z()
    U0Y = Z(); U1Y = Z(); VDY = Z(); VY = Z()
    U0Z = Z(); U1Z = Z(); VDZ = Z(); VZ = Z()

    FLX = Z(); FLY = Z(); FLZ = Z()

    # Random forces
    for X in range(30, 60):
        for Y in range(LY + 8):
            FLX[Y, X] = F0 * SQM * np.cos(2*np.pi*np.random.rand())
            FLY[Y, X] = F0 * SQM * np.cos(2*np.pi*np.random.rand())
            FLZ[Y, X] = F0 * SQM * np.cos(2*np.pi*np.random.rand())

    # Time evolution
    for N in range(ECST + 1):
        ct = np.cos(W * N * TAU)

        for X in range(30, 60):
            for Y in range(3, LY + 5):

                if M[Y, X] <= 0:
                    continue

                VDX[Y, X] = FLX[Y, X] * ct
                VDY[Y, X] = FLY[Y, X] * ct
                VDZ[Y, X] = FLZ[Y, X] * ct

                if (X % 3 == 2) and (Y % 2 == 1):
                    VDX[Y, X] += (
                        Kra2[Y, X]  * (U0X[Y+1, X] - U0X[Y, X]) * 0.5 +
                        Kin2[Y, X]  * (U0X[Y+1, X] - U0X[Y, X]) * (np.sqrt(3)/2)
                    )

                # ⚠️ Remaining force terms continue here
                # Paste remaining MATLAB lines to complete conversion

# --------------------------------------------------
print(f"Execution time: {time.time() - t0:.2f} s")
