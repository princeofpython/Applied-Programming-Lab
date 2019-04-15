import numpy as np
import cmath
class Resistor(object):
    def __init__(self,n1,n2,value):
        self.n1=n1
        self.n2=n2
        self.value=float(value)
        self.impedance=float(value)

class ACVsource(object):
    def __init__(self,n1,n2,value,phase):
        self.n1=n1
        self.n2=n2
        self.value=cmath.rect(float(value)/(2.828),(float(phase))*(np.pi/180))
        self.phase=float(phase)
class DCVsource(object):
    def __init__(self,n1,n2,value):
        self.n1=n1
        self.n2=n2
        self.value=float(value)
class ACCsource(object):
    def __init__(self,n1,n2,value,phase):
        self.n1=n1
        self.n2=n2
        self.value=cmath.rect(float(value)/2.828,(float(phase))*(np.pi/180))
        self.phase=float(phase)
class DCCsource(object):
    def __init__(self,n1,n2,value):
        self.n1=n1
        self.n2=n2
        self.value=float(value)
class Capacitor(object):
    def __init__(self,n1,n2,value):
        self.n1=n1
        self.n2=n2
        self.value=float(value)
        self.impedance=1/complex(0,freq*2*np.pi*self.value)
class Inductor(object):
    def __init__(self,n1,n2,value):
        self.n1=n1
        self.n2=n2
        self.value=float(value)
        self.impedance=complex(0,freq*2*np.pi*self.value)
import sys
try:
    arg=sys.argv[1]
except IndexError:
    print("Argument isn't given")
    quit()
if len(sys.argv)>2:
    print('More Arguments detected')
    quit()
try:
    with open(arg) as f:
        lines=f.read().splitlines()
except:
    print("Could not read file")
    quit()
firstlines=[]
for line in lines:
    firstlines.append(line.split()[0])
circuitlines=[]
if ".circuit" not in firstlines:
    print("No dot command for circuit")
    quit()
if ".end" not in firstlines:
    print("No dot command for end")
    quit()
if firstlines.index(".circuit")>firstlines.index(".end"):
    print("Invalid circuit format")
    quit()
nodes={"GND":0}
nodenum=1
sources=0
circuitlist=[]
for n in range(len(lines)):
    if lines[n][:8]==".circuit":
        for m in range(n+1,len(lines)):
            if(lines[m][:4]!=".end"):
                lines[m]=lines[m].split("#",1)[0]
                splitted=lines[m].split()
                circuitlines.append(splitted)                
            else:
                break
        try:
            while(True):
                if lines[m+1][:3]==".ac":
                    splitted=lines[m+1].split()
                    global freq
                    freq=float(splitted[2])
                    break
                m+=1
        except:
            pass
        break
acflag=0
for splitted in circuitlines: 
    if splitted[1] not in nodes:
        nodes[splitted[1]]=nodenum
        nodenum+=1
    if splitted[2] not in nodes:
        nodes[splitted[2]]=nodenum
        nodenum+=1
for splitted in circuitlines:
    if splitted[0][0]=="R":
        circuitlist.append(Resistor(nodes[splitted[1]],nodes[splitted[2]],splitted[3]))
    if splitted[0][0]=="C":
        circuitlist.append(Capacitor(nodes[splitted[1]],nodes[splitted[2]],splitted[3]))
    if splitted[0][0]=="L":
        circuitlist.append(Inductor(nodes[splitted[1]],nodes[splitted[2]],splitted[3]))
    if splitted[0][0]=="V" and splitted[3]=="dc":
        circuitlist.append(DCVsource(nodes[splitted[1]],nodes[splitted[2]],splitted[4]))
        sources+=1
    if splitted[0][0]=="V" and splitted[3]=="ac":
        circuitlist.append(ACVsource(nodes[splitted[1]],nodes[splitted[2]],splitted[4],splitted[5]))
        sources+=1
        acflag=1
    if splitted[0][0]=="I" and splitted[3]=="dc":
        circuitlist.append(DCCsource(nodes[splitted[1]],nodes[splitted[2]],splitted[4]))      
        sources+=1
    if splitted[0][0]=="I" and splitted[3]=="ac":
        circuitlist.append(ACCsource(nodes[splitted[1]],nodes[splitted[2]],splitted[4],splitted[5]))
        sources+=1
        acflag=1
n=len(nodes)
lenn=sources+n
Matx=np.zeros((lenn,lenn),dtype=complex)
Maty=np.zeros((lenn,),dtype=complex)
for obj in circuitlist:
    source=n
    if isinstance(obj,Resistor) or isinstance(obj,Inductor) or isinstance(obj,Capacitor):
        Matx[obj.n1,obj.n1]+=1/obj.impedance
        Matx[obj.n2,obj.n2]+=1/obj.impedance
        Matx[obj.n1,obj.n2]-=1/obj.impedance
        Matx[obj.n2,obj.n1]-=1/obj.impedance
    if isinstance(obj,ACVsource) or isinstance(obj,DCVsource):
        Matx[source][obj.n1]=1
        Matx[source][obj.n2]=-1
        Matx[obj.n1][source]=1
        Matx[obj.n1][source]=-1
        Maty[source]=obj.value
        source+=1
Matx[0]=np.zeros((lenn,),dtype=complex)
Matx[0][0]=1+0j
ans=np.linalg.solve(Matx,Maty)
if acflag==0:
    for each in ans:
        print(each.real)
else:
    for each in ans:
        print(each)
        






