#from IPython import get_ipython
#get_ipython().magic('reset -sf')
import numpy as np
import matplotlib.pyplot as plt

# All functions related to Linear equation 
class Linear:
    flux_name = 'Linear' 
    def __init__(self):
        def flux(self,u):
            # true flux
            return u
        self.__class__.flux = flux 
        def fluxd(self,u):
            # flux derivative
            return 1
        self.__class__.fluxd = fluxd
        def ec_flux(self,ul,ur):
            # entropy conservative flux
            return 0.5*(ul+ul)
        self.__class__.ec_flux = ec_flux
        
# All functions related to Burger equation         
class Burger:
    flux_name = 'Burger' 
    def __init__(self):
        def flux(self,u):
            # true flux
            return u**2/2.0
        self.__class__.flux = flux 
        def fluxd(self,u):
            # flux derivative
            return u
        self.__class__.fluxd = fluxd
        def ec_flux(self,ul,ur):
            # entropy conservative flux
            return (1.0/6.0)*(ul**2+ul*ur+ur**2)
        self.__class__.ec_flux = ec_flux

# All test problems        
class Test_Problem:
    global N
    def __init__(self,test):
        if test==1:
            self.a,self.b=-1,1
            self.x=np.linspace(self.a,self.b,N+1)
            self.dx=(self.b-self.a)/N
            self.u0=-np.sin(np.pi*self.x)
            self.T=2
        elif test==2:
            self.a,self.b=-1,1
            self.x=np.linspace(self.a,self.b,N+1)
            self.dx=(self.b-self.a)/N
            self.u0=(np.abs(self.x)<1/3.0)*1+(np.abs(self.x)>=1/3.0)*(0)
            self.T=0.5
        elif test==3:
            self.a,self.b=-1,1
            self.x=np.linspace(self.a,self.b,N+1)
            self.dx=(self.b-self.a)/N
            self.u0=(1+0.5*np.sin(np.pi*self.x))
            self.T=1.5   
        elif test==4:
            self.a,self.b=-1,1
            self.x=np.linspace(self.a,self.b,N+1)
            self.dx=(self.b-self.a)/N
            self.u0=(self.x<-1/3.0)*1+(self.x>=-1/3.0)*(0)
            self.T=0.5 
            
def add_ghost_cells(ub,bc,gcn):
    # This function will add gcn number of ghost cells at boundary
    if bc=='Periodic':
        ub=np.insert(ub,N+1,ub[1:gcn+1])
        ub=np.insert(ub,0,ub[N-gcn:N])
    elif bc=='Neumann':
        ub=np.insert(ub,N+1,gcn*[ub[-1]])
        ub=np.insert(ub,0,gcn*[ub[0]])
    return ub

def time_evolution(u0,dt):
    # This function define the time integration procedure
    methodd='3rdSSP'
    if methodd=='EulerForward':
        u1=u0-dt*Lu(u0)
    elif methodd=='3rdSSP':
        u_1=u0-dt*Lu(u0)
        u_2=(3/4)*u0+(1/4)*u_1-(1/4)*dt*Lu(u_1)
        u1=(1/3)*u0+(2/3)*u_2-(2/3)*dt*Lu(u_2)
    return u1

def entropy_stable_flux(u):
# This function calculate the entropy stable flux
    global cl, bc, gcn, N
    ub=add_ghost_cells(u,bc,gcn)
    # ___ulll___ull___ul_(*)_ur___urr___urrr___
    ulll=ub[0:N+2]
    ull=ub[1:N+3]
    ul=ub[2:N+4]
    ur=ub[3:N+5]
    urr=ub[4:N+6]
    urrr=ub[5:N+7]
    
    fc1=cl.ec_flux(ul,ur) # second order entropy conservative flux
    fc2=cl.ec_flux(ull,ur)
    fc3=cl.ec_flux(ul,urr)
    
    fc=fc1*(4.0/3.0)-(1.0/6.0)*(fc2+fc3); # Fourth order entropy conservative flux
    af=np.max(np.abs(cl.fluxd(u))) 
    
    ul_rec=np.zeros(N+2)
    ur_rec=np.zeros(N+2)
    for i in range(0,N+2):
        # reconstructions at i-1/2
        ul_rec[i],ur_rec[i]=eno3_interpolation(ulll[i],ull[i],ul[i],ur[i],urr[i],urrr[i])
        if (ur_rec[i]-ul_rec[i])*(ul[i]-ur[i])>1e-14:
            print('Reconstruction is not sign preserving')
    fl=fc-0.5*af*(ur_rec-ul_rec)
    return fl
    
def numerical_flux(u0):
    fs=entropy_stable_flux(u0)
    return fs        
def Lu(u0):
    global dx, N
    fl=numerical_flux(u0)
    return (fl[1:N+2]-fl[0:N+1])/dx

def eno3_interpolation_left(um2,um1,u,up1,up2,oprtr):
    # This function only caculate u_{i-1/2}^{-}
    # Same function will be used to calculate u_{i-1/2}^{+}
    ENO3_COEF= [[0.375,-1.25,1.875],[-0.125,0.75,0.375],[0.375,0.75,-0.125]] #ENO coefficient generally denoted as c_{ij} 
    state=[um2, um1, u, up1, up2]; # input vector
    eno_level=3;
    ul=0;
    # Calculation of undivided difference
    dd=np.zeros([eno_level,2*eno_level-1])
    for j in range(0,2*eno_level-1):
        dd[0,j] = state[j]
    for i in range(1,eno_level):
        for j in range(0,2*eno_level-1-i):
            dd[i,j] = dd[i-1,j+1] - dd[i-1,j]   
    r=0;
    s = eno_level-1;
    # Calculation of smoothness indicator r
    for i in range(1,eno_level):
        if oprtr(abs(dd[i,s-r]), abs(dd[i,s-r-1])):
            r+=1
    # Finally calculate the reconstruction sum(c_{ij}u_{..}) 
    for i in range(0,eno_level):
        ul+= ENO3_COEF[s-r][i]*state[s-r+i];
    return ul
def eno3_interpolation(um3,um2,um1,u,up1,up2):
    # This is the main function which will take 6 input values
    # and provide you the reconstructed values u_{i-1/2}^{\pm}   
    ul=eno3_interpolation_left(um3,um2,um1,u,up1,operator.gt) # operator.gt is used to do the comparison of divided difference
    ur=eno3_interpolation_left(up2,up1,u,um1,um2,operator.ge) # if operator.gt is from left then operator.ge is from right
    return ul,ur
## Main Code
N=200 # number of cells
bc='Periodic';gcn=3 # Initialize boundary condition and number of ghost cells. options(bc): 'Periodic','Neumann'
tp=Test_Problem(1)  # Call the test problem. options: 1,2,3,4
cl=Burger()         # Call the ConservationLaw. options: 'Linear', 'Burger'
CFL=0.5
u0=tp.u0
dx=tp.dx
t=0
while t<tp.T:
    af=np.max(np.abs(cl.fluxd(tp.u0))) 
    dt=CFL*dx/af
    if dt>tp.T-t:
        dt=tp.T-t
    u1=time_evolution(u0,dt)
    t=t+dt
    u0=u1
fig=plt.figure()
plt.plot(tp.x,u1,'-*')
plt.show()
