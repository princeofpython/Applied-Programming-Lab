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
circuitlines=[]
if ".circuit" not in lines:
    print("No dot command for circuit")
    quit()
if ".end" not in lines:
    print("No dot command for end")
    quit()
if lines.index(".circuit")>lines.index(".end"):
    print("Invalid circuit format")
    quit()
for n in range(len(lines)):
    if lines[n]==".circuit":
        for m in range(n+1,len(lines)):
            if(lines[m]!=".end"):
                circuitlines.append(lines[m])
            else:
                break
        break
circuitlines.reverse()
for k in circuitlines:
    if '#' in k:
        k=k.split("#",1)[0]
    kn=k.split()
    kn.reverse()
    print((' ').join(kn))
    
    
    
