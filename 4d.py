import math,pygame,time,random
import numpy as np
from numba import vectorize,float64,int64,uint8,guvectorize,uint32

clock = pygame.time.Clock()
width=1300
height=1300
print('Resolution : '+str(width)+ " " + str(height))
sca=1
screen = pygame.display.set_mode((width*sca,height*sca))
clock = pygame.time.Clock()
update,done=1,False

def vec(x):
    return np.array(x,dtype = np.float64)
def vec4(a,b,c,d):
    return np.array([a,b,c,d],dtype = np.float64)
def vec3(a,b,c):
    return np.array([a,b,c],dtype = np.float64)

def makecircles(N,reps):
    global circles, n
    n =N
    circles = list(range(reps))
    for x in range (reps):
        circles[x] = np.zeros((n,4),dtype = np.float64)
    return circles
def makeunitcircles():
    n = 20
    makecircles(n,3)
    for x in range(n):
        a = x / n * math.pi*2+math.pi/2
        circles[0][x][0] = np.sin(a)
        circles[0][x][1] = np.cos(a)
        
    for x in range(n):
        a = x / n * math.pi*2+math.pi/2
        circles[1][x][2] = np.sin(a)
        circles[1][x][3] = np.cos(a)
    for x in range(n):
        a = x / n * math.pi*2+math.pi/2
        if x%2==0:
            circles[2][x][2] = np.sin(a)
            circles[2][x][3] = np.cos(a)
        else:
            circles[2][x][0] = np.sin(a)
            circles[2][x][1] = np.cos(a)

def addtocircles(new):
    newreps = len(new)
    newN = new[0].shape[0]
    global circles
    circles = circles + new
global circles,n
n=3000
circles = []

def makeoctahedron():
    makecircles(4,6)
    w = 0
    comb = [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
    ns = 0
    for c in comb:
        #print(c)
        for x in range(n):
            a = x / n * math.pi*2+math.pi/2
            #point = vec4(0,0,0,0)
            #points[ns][c[0]-1] = np.sin(a)
            #points[ns][c[1]-1] = np.cos(a)
            circles[ns][x][c[0]-1] = np.sin(a)
            circles[ns][x][c[1]-1] = np.cos(a)
        ns+=1
    return circles

def makeoctahedronb():
    #makecircles(4,6)
    w = 0

    
    comb = np.array([[1,1,1,1],[-1,1,1,1],[1,-1,1,1],[-1,-1,1,1],[1,1,-1,1],[-1,1,-1,1],[1,-1,-1,1],[-1,-1,-1,1],[1,1,1,-1],[-1,1,1,-1],[1,-1,1,-1],[-1,-1,1,-1],[1,1,-1,-1],[-1,1,-1,-1],[1,-1,-1,-1],[-1,-1,-1,-1]])
    
    ggh = np.array([[1,1,1,0],[0,1,1,1],[1,0,1,1],[1,1,0,1]])
    makecircles(4,len(comb))
    ns = 0
    for c in comb:
        #print(c)
        for x in range(4):
            #a = x / n * math.pi*2+math.pi/2
            #point = vec4(0,0,0,0)
            #points[ns][c[0]-1] = np.sin(a)
            #points[ns][c[1]-1] = np.cos(a)
            circles[ns][x]= c*ggh[x]
            
        ns+=1
    return circles

def makeoctahedron():
    #makecircles(4,6)
    w = 0
    
    
    comb = np.array([[1,1,1,0],[-1,1,1,0],[1,-1,1,0],[-1,-1,1,0],[1,1,-1,0],[-1,1,-1,0],[1,-1,-1,0],[-1,-1,-1,0],[1,1,0,1],[-1,1,0,1],[1,-1,0,1],[-1,-1,0,1],[1,1,0,-1],[-1,1,0,-1],[1,-1,0,-1],[-1,-1,0,-1],[1,0,1,1],[-1,0,1,1],[1,0,-1,1],[-1,0,-1,1],[1,0,1,-1],[-1,0,1,-1],[1,0,-1,-1],[-1,0,-1,-1],[0,1,1,1],[0,-1,1,1],[0,1,-1,1],[0,-1,-1,1],[0,1,1,-1],[0,-1,1,-1],[0,1,-1,-1],[0,-1,-1,-1]])
    
    ggh = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    makecircles(3,len(comb))
    ns = 0
    for c in comb:
        #print(c)
        x = 0
        x1 = 0
        while x<4:
            #a = x / n * math.pi*2+math.pi/2
            #point = vec4(0,0,0,0)
            #points[ns][c[0]-1] = np.sin(a)
            #points[ns][c[1]-1] = np.cos(a)
            if (c*ggh[x] == [0,0,0,0]).all():
                pass
            else:
                circles[ns][x1]= c*ggh[x]
                x1+=1
            x+=1
            
            
        ns+=1
    return circles
    
def makeoctahedronc():
    #makecircles(4,6)
    w = 0
    
    
    comb = np.array([[1,1,1,0],[-1,1,1,0],[1,-1,1,0],[-1,-1,1,0],[1,1,-1,0],[-1,1,-1,0],[1,-1,-1,0],[-1,-1,-1,0],[1,1,0,1],[-1,1,0,1],[1,-1,0,1],[-1,-1,0,1],[1,1,0,-1],[-1,1,0,-1],[1,-1,0,-1],[-1,-1,0,-1],[1,0,1,1],[-1,0,1,1],[1,0,-1,1],[-1,0,-1,1],[1,0,1,-1],[-1,0,1,-1],[1,0,-1,-1],[-1,0,-1,-1],[0,1,1,1],[0,-1,1,1],[0,1,-1,1],[0,-1,-1,1],[0,1,1,-1],[0,-1,1,-1],[0,1,-1,-1],[0,-1,-1,-1]])
    
    ggh = 1-np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    makecircles(3,len(comb))
    ns = 0
    for c in comb:
        #print(c)
        x = 0
        x1 = 0
        while x<4:
            #a = x / n * math.pi*2+math.pi/2
            #point = vec4(0,0,0,0)
            #points[ns][c[0]-1] = np.sin(a)
            #points[ns][c[1]-1] = np.cos(a)
            if np.linalg.norm(c*ggh[x]) > 1.5:
                pass
            else:
                circles[ns][x1]= c*ggh[x]*5
                x1+=1
            x+=1
            
            
        ns+=1
    return circles

def make5cell():
    
    points = np.zeros((5,4),dtype=np.float64)
    for x in range(4):
        points[x][x]=2
    points[4] = vec4(1,1,1,1)*(1+5**.5)/2
    comb = [[0,1,2],[0,1,3],[0,1,4],
    [0,2,3],[0,2,4],
    [0,3,4],
    
    ]
    makecircles(3,len(comb))
    i = 0
    for x in comb:
        for g in range(3):
            circles[i][g]=points[x[g]]
        i+=1
    mid = np.average(points,axis=0)
    for x in circles:
        x-=mid
        x*=.5

def makehypercube():
    makecircles(4,24)
    circles[0] = np.array([[0,0,0,0],[0,0,0,1],[0,0,1,1],[0,0,1,0]],dtype=np.float64)
    circles[1] = circles[0]+vec4(1,0,0,0)
    circles[2] = circles[0]+vec4(1,1,0,0)
    circles[3] = circles[0]+vec4(0,1,0,0)

    circles[4] = np.array([[0,0,0,0],[0,0,0,1],[1,0,0,1],[1,0,0,0]],dtype=np.float64)
    circles[5] = circles[4]+vec4(0,1,0,0)
    circles[6] = circles[4]+vec4(0,1,1,0)
    circles[7] = circles[4]+vec4(0,0,1,0)

    circles[8] = np.array([[0,0,0,0],[0,1,0,0],[1,1,0,0],[1,0,0,0]],dtype=np.float64)
    circles[9] = circles[8]+vec4(0,0,1,0)
    circles[10] = circles[8]+vec4(0,0,1,1)
    circles[11] = circles[8]+vec4(0,0,0,1)

    circles[12] = np.array([[0,0,0,0],[0,0,1,0],[0,1,1,0],[0,1,0,0]],dtype=np.float64)
    circles[13] = circles[12]+vec4(0,0,0,1)
    circles[14] = circles[12]+vec4(1,0,0,1)
    circles[15] = circles[12]+vec4(1,0,0,0)

    circles[16] = np.array([[0,0,0,0],[0,1,0,0],[0,1,0,1],[0,0,0,1]],dtype=np.float64)
    circles[17] = circles[16]+vec4(1,0,0,0)
    circles[18] = circles[16]+vec4(1,0,1,0)
    circles[19] = circles[16]+vec4(0,0,1,0)

    circles[20] = np.array([[0,0,0,0],[1,0,0,0],[1,0,1,0],[0,0,1,0]],dtype=np.float64)
    circles[21] = circles[20]+vec4(0,1,0,0)
    circles[22] = circles[20]+vec4(0,1,0,1)
    circles[23] = circles[20]+vec4(0,0,0,1)
    for c in circles:
        c-=.5
    return circles

def makehyperrect():
    makecircles(4,48)
    circles[0] = np.array([[0,0,0,0],[0,0,0,1],[0,0,1,1],[0,0,1,0]],dtype=np.float64)
    circles[1] = circles[0]+vec4(1,0,0,0)
    circles[2] = circles[0]+vec4(1,1,0,0)
    circles[3] = circles[0]+vec4(0,1,0,0)

    circles[4] = np.array([[0,0,0,0],[0,0,0,1],[1,0,0,1],[1,0,0,0]],dtype=np.float64)
    circles[5] = circles[4]+vec4(0,1,0,0)
    circles[6] = circles[4]+vec4(0,1,1,0)
    circles[7] = circles[4]+vec4(0,0,1,0)

    circles[8] = np.array([[0,0,0,0],[0,1,0,0],[1,1,0,0],[1,0,0,0]],dtype=np.float64)
    circles[9] = circles[8]+vec4(0,0,1,0)
    circles[10] = circles[8]+vec4(0,0,1,1)
    circles[11] = circles[8]+vec4(0,0,0,1)

    circles[12] = np.array([[0,0,0,0],[0,0,1,0],[0,1,1,0],[0,1,0,0]],dtype=np.float64)
    circles[13] = circles[12]+vec4(0,0,0,1)
    circles[14] = circles[12]+vec4(1,0,0,1)
    circles[15] = circles[12]+vec4(1,0,0,0)

    circles[16] = np.array([[0,0,0,0],[0,1,0,0],[0,1,0,1],[0,0,0,1]],dtype=np.float64)
    circles[17] = circles[16]+vec4(1,0,0,0)
    circles[18] = circles[16]+vec4(1,0,1,0)
    circles[19] = circles[16]+vec4(0,0,1,0)

    circles[20] = np.array([[0,0,0,0],[1,0,0,0],[1,0,1,0],[0,0,1,0]],dtype=np.float64)
    circles[21] = circles[20]+vec4(0,1,0,0)
    circles[22] = circles[20]+vec4(0,1,0,1)
    circles[23] = circles[20]+vec4(0,0,0,1)
    
    for x in range(0,24):
        circles[x+24]=circles[x]+vec4(0,1,0,0)
    
    circles[0] = np.array([[0,0,0,0],[0,0,0,1],[0,0,1,1],[0,0,1,0]],dtype=np.float64)
    circles[1] = circles[0]+vec4(1,0,0,0)
    circles[2] = circles[0]+vec4(1,1,0,0)
    circles[3] = circles[0]+vec4(0,1,0,0)

    circles[4] = np.array([[0,0,0,0],[0,0,0,1],[1,0,0,1],[1,0,0,0]],dtype=np.float64)
    circles[5] = circles[4]+vec4(0,1,0,0)
    circles[6] = circles[4]+vec4(0,1,1,0)
    circles[7] = circles[4]+vec4(0,0,1,0)

    circles[8] = np.array([[0,0,0,0],[0,1,0,0],[1,1,0,0],[1,0,0,0]],dtype=np.float64)
    circles[9] = circles[8]+vec4(0,0,1,0)
    circles[10] = circles[8]+vec4(0,0,1,1)
    circles[11] = circles[8]+vec4(0,0,0,1)

    circles[12] = np.array([[0,0,0,0],[0,0,1,0],[0,1,1,0],[0,1,0,0]],dtype=np.float64)
    circles[13] = circles[12]+vec4(0,0,0,1)
    circles[14] = circles[12]+vec4(1,0,0,1)
    circles[15] = circles[12]+vec4(1,0,0,0)

    circles[16] = np.array([[0,0,0,0],[0,1,0,0],[0,1,0,1],[0,0,0,1]],dtype=np.float64)
    circles[17] = circles[16]+vec4(1,0,0,0)
    circles[18] = circles[16]+vec4(1,0,1,0)
    circles[19] = circles[16]+vec4(0,0,1,0)

    circles[20] = np.array([[0,0,0,0],[1,0,0,0],[1,0,1,0],[0,0,1,0]],dtype=np.float64)
    circles[21] = circles[20]+vec4(0,1,0,0)
    circles[22] = circles[20]+vec4(0,1,0,1)
    circles[23] = circles[20]+vec4(0,0,0,1)
    
    for c in circles:
        c-=.5
    return circles

def makehyperdiscs():
    N=400
    makecircles(N,3)
    
    for x in range(N):
        a=x/N*math.pi*20
        t=x/100
        circles[0][x][0]=np.sin(a)
        circles[0][x][1]=np.cos(a)
    for x in range(N):
        a=x/N*math.pi*20
        t=x/100
        circles[1][x][2]=np.sin(a)
        circles[1][x][3]=np.cos(a)
    for x in range(N):
        a=x/N*math.pi/2
        circles[2][x] = np.sin(a)*circles[1][x]+np.cos(a)*circles[0][x]

def makeabunchofcircles():
    makecircles(600,1)
    w = 0
    comb = [[1,2,3],[1,2,4],[1,3,4],[2,3,4]]
    ns = 0
    g = 5
    for x in range(n):
        a = x / n * math.pi*2

        circles[0][x][3] = np.sin(a*g)*np.cos(a)*np.sin(a*6*g)
        circles[0][x][1] = np.cos(a*g)*np.cos(a)*np.sin(a*6*g)
        circles[0][x][2] = np.sin(a)*np.sin(a*6*g)
        circles[0][x][0] = np.cos(a)
        circles[0][x] /= np.linalg.norm(circles[0][x])

def makehypercubenew():
    makecircles(4,24)
    o = np.array([0,0,0,0])
    
    n = 0
    for x in range(8):
        pos = 1-(x%2)*2
        
        mas = np.array ( [x<2,2<=x<4,4<=x<6,6<=x<8])
        
        for m in range(6):
            for i in range(4):
                add = vec([0,0,0,0])
                add[(m+1+x//2)%4] = 1-(i>=2)*2
                add[(m+(m==2)+2+x//2)%4] = 1-(3>i>=1)*2
                circles[n][i] = (mas*pos + add)/2
                #print(mas*pos + add)
            
            n+=1

def makehypercubenew():
    makecircles(4,24)
    o = np.array([0,0,0,0])
    
    n = 0
    for x in range(6):
        pos = 1-(x%2)*2
        
        
        ns = [(x%4),(x+(x>=4)+1)%4,(x+2-(x>=4))%4,(x+3)%4]
        print(ns)
        for m in range(4):
            pos = 1-(x%2)*2
            mas = o+0
            mas[ns[0]]= 1-(m%2)*2
            mas[ns[1]]= 1-((m//2)%2)*2
            for i in range(4):
                add = vec([0,0,0,0])
                add[ns[2]] = 1-(i>=2)*2
                add[ns[3]] = 1-(3>i>=1)*2
                circles[n][i] = (mas*pos + add)/2
                #print(mas*pos + add)
            
            n+=1
def makehypercubenew():
    
    makecircles(4,24)

    for x in range(6):
        ns = [(x%4),(x+(x>=4)+1)%4,(x+2-(x>=4))%4,(x+3)%4]
        for m in range(4):
            mas = vec([0,0,0,0])
            mas[ns[0]]= m%2
            mas[ns[1]]= (m//2)%2
            for i in range(4):
                add = vec([0,0,0,0])
                add[ns[2]] = i>=2
                add[ns[3]] = 3>i>=1
                circles[m+x*4][i] = (mas + add) -  vec([1,1,1,1])/2

def makehypercubenew():
    
    makecircles(4,24)
    global circles
    hcube = circles[0:24]+np.array([0])
    for x in range(6):
        ns = [(x%4),(x+(x>=4)+1)%4,(x+2-(x>=4))%4,(x+3)%4]
        for m in range(4):
            mas = vec([0,0,0,0])
            mas[ns[0]]= m%2
            mas[ns[1]]= (m//2)%2
            hcube[m+x*4][:]+=mas
            for i in range(4):
                add = vec([0,0,0,0])
                add[ns[2]] = i>=2
                add[ns[3]] = 3>i>=1
                hcube[m+x*4][i] += (add)
    hcube -= vec([1,1,1,1])/2
    circles[0:24]= hcube*1.5

def makehypercubenew4():
    
    makecircles(4,219)
    global circles
    hcube = circles[0:24]+np.array([0])
    for x in range(6):
        ns = [(x%4),(x+(x>=4)+1)%4,(x+2-(x>=4))%4,(x+3)%4]
        for m in range(4):
            mas = vec([0,0,0,0])
            mas[ns[0]]= m%2
            mas[ns[1]]= (m//2)%2
            hcube[m+x*4][:]+=mas
            for i in range(4):
                add = vec([0,0,0,0])
                add[ns[2]] = i>=2
                add[ns[3]] = 3>i>=1
                hcube[m+x*4][i] += (add)
    i=0
    for x in range(2):
        for y in range(2):
            for z in range(2):
                for w in range(2):
                    
                    cube_to_add = (hcube+vec([x,y,z,w])-vec([1,1,1,1]))/2
                    
                    
                    for c in cube_to_add:
                        flag = True
                        avc = np.average(c,0)
                        
                        if np.linalg.norm(c) < 1.7:
                             flag = False
                        
                        if flag:
                            avC_arr = np.average(circles,1)
                            for C in range(0,i):
                                
                                # print(avC_arr[])
                                
                                if (avC_arr[C] == avc).all():
                                    flag = False
                                    break
                                    # sprint("flag: false")
                        if flag:
                            circles[i] = c+0
                            i+=1

    print (i, "faces, reduced from 384")
                    
    
    # hcube -= vec([1,1,1,1])/2
    # circles[0:24]= hcube*1.5
    # circles[0:24]= hcube+0
    # n= 0
    # for x in range(16):
    #     add = vec([x%2, (x//2)%2,(x//4)%2,(x//8)%2])
    #     print(add,add.sum())
    #     if add.sum()%2 ==0:
    #         print('true')
    #         circles[24+24*n:48+24*n] = hcube*.1+add-vec([1,1,1,1])/2
    #         n+=1
        
    

shapefunclist = [makeabunchofcircles,
makehypercube,
makehyperrect,
makehyperdiscs,
makeoctahedron,
makeoctahedronb,
make5cell,
makeunitcircles,
makehypercubenew,
makehypercubenew4,
]

shapefunclistnames = ['makeabunchofcircles',
'makehypercube',
'makehyperrect',
'makehyperdiscs',
'makeoctahedron',
'makeoctahedronb',
'make5cell',
'makeunitcircles',
'makehypercubenew',
'makehypercubenew4',
    ]

global shape_number
shape_number = 9
shapefunclist[shape_number]()
#temp = circles
#shapefunclist[1]()
#addtocircles(temp)

###################################################make objects#####################################

def rotcub(cub,t=1,dim1=0,dim2=3):
    a = .01*t
    for c in range(len(cub)):
        temp= np.swapaxes(cub[c],0,1)
        temp2 = temp + 0
        #print(temp)
        temp2[dim1] =temp[dim1]*np.cos(a) - temp[dim2]*np.sin(a)
        temp2[dim2] = temp[dim1]*np.sin(a) + temp[dim2]*np.cos(a)
       #print(temp2, "222")
        cub[c] = np.swapaxes(temp2,0,1)
        #cub[c] = cub[c]*vec4(np.cos(a),1,1,np.sin(a))+cub[c]*vec4(-np.sin(a),0,0,np.cos(a))
    return cub

class O:
    def __init__(self,v=vec([1,0,0,0]),sscoord=vec([0,0,0,0]),l=vec([0,0,0,0])):
        self.v = v
        self.l = l
        self.ss=sscoord

def supersphere(ss):
    xym = np.cos(ss[0])
    zwm = np.sin(ss[0])
    return vec4(
        xym*np.cos(ss[1]),
        xym*np.sin(ss[1]),
        zwm*np.cos(ss[2]),
        zwm*np.sin(ss[2])
    )
def mag2(a,b):
    return np.sqrt(a*a+b*b)

def tosupersphere(p):
    xytzw = np.arctan2(mag2(p[2],p[3]),mag2(p[0],p[1]))
    xty = np.arctan2(p[1],p[0])
    ztw = np.arctan2(p[3],p[2])
    return vec3(xytzw,xty,ztw)
o = O()

def ui():
    elc = [width - 300,height - 75]
    els = [150, 100]
    pygame.draw.ellipse(screen,(200,200,200),
        (elc[0]-els[0]/2,elc[1]-els[1]/2,els[0],els[1])
    )
    a = o.ss[1]
    pygame.draw.line(screen,(0,100,250),
        (elc[0],elc[1]),
        (els[0]*np.cos(a)/2+elc[0],els[1]*np.sin(a)/2+elc[1]),3
    )
    
    elc = [width - 100,height - 75]
    els = [100, 150]
    pygame.draw.ellipse(screen,(200,200,200),
        (elc[0]-els[0]/2,elc[1]-els[1]/2,els[0],els[1])
    )
    a = o.ss[2]
    pygame.draw.line(screen,(0,100,250),
        (elc[0],elc[1]),
        (els[0]*np.cos(a)/2+elc[0],els[1]*np.sin(a)/2+elc[1]),3
    )
    elc = [100,height - 75]
    els = [100, 150]
    pygame.draw.ellipse(screen,(200,200,200),
        (elc[0]-els[0]/2,elc[1]-els[1]/2,els[0],els[1])
    )
    a = o.ss[0]
    pygame.draw.line(screen,(0,100,250),
        (elc[0],elc[1]),
        (els[0]*np.cos(a)/2+elc[0],els[1]*np.sin(a)/2+elc[1]),3
    )
    textsurface = myfont.render(str(o.ss), False, (250, 250, 250))
    screen.blit(textsurface,(0,0))
    textsurface = myfont.render(str(supersphere(o.ss)), False, (250, 250, 250))
    screen.blit(textsurface,(0,30))
    #############
    ss = np.degrees(o.ss)


    left = supersphere(leftsupersphere(ss))
    tex(left,60)

    
    up = supersphere(upsupersphere(ss))
    tex(up,90)
    tex(up.dot(left),120)


def tex(st,h):
    textsurface = myfont.render(str(st), False, (250, 250, 250))
    screen.blit(textsurface,(0,h))

def perspective(p):
    P=p-o.l

    ss = np.degrees(o.ss) #shit shit shit i am shittttttingng

    v= supersphere(o.ss)

    left = supersphere(leftsupersphere(ss))
   
    up = supersphere(upsupersphere(ss))
    
    D =P.dot(v)

    W = P.dot(left)/(D)
    H = P.dot(up)/(D)
    return(W,H,D)

#print(perspective(points))
def upsupersphere(ss):
    return np.radians(vec3(90+ss[0],ss[1],ss[2]))

def leftsupersphere(ss):
    return np.radians(vec3(0,ss[1]-90,0))


def show():
    c = 0
    cl = len(circles)
    for points in circles:
        W,H,D = perspective(points)
        screenpoints = np.column_stack((W,H))
        mask = D>.1

        screenpoints = screenpoints[mask]
        D=D[mask]
        #print(screenpoints)
        screenpoints*=800
        #print(screenpoints)
        screenpoints += vec([width/2,height/2])
        #print(screenpoints[0])
        D = (10/D).astype(np.int64)
        screenpoints = screenpoints.astype(np.int64)
        if (screenpoints.size>2):
            pygame.draw.aalines(screen,(100+100*np.sin(c),255-c*250/cl,c*250/cl),True,screenpoints,9)
            for p in range(screenpoints.shape[0]):
                pygame.draw.circle(screen,
                (120+120*np.sin(c*1+4),120+120*np.sin(c*2),120+120*np.sin(-c*2)),
                screenpoints[p],D[p])
        c+=1

def run_shader():
    c = 0
    cl = len(circles)
    
    for points in circles:
        W,H,D = perspective(points)
        screenpoints = np.column_stack((W,H,D))
        

        screenpoints *= vec3(800,800,1)
        screenpoints += vec3(width/2,height/2,0)
        for x in range(screenpoints.shape[0]):
            # print(c,x,'c,x')
            # print(verts.shape,'vertshape')
            # print(screenpoints.shape,'screenshape')
            verts[c][x]=screenpoints[x]
        
        c+=1
    
    tops = gettops(coors,verts)

    return usetops(coors,coorsy,tops)
    
##################################################polygons start

coors = np.arange(0,width,1,dtype = np.float64)

@guvectorize([(float64,float64[:,:,:],float64[:,:,:])],'(),(a,t,b)->(a,t,b)')
def gettops(X,tris,res):
    h = height
    res *= 0
    for tr in range(tris.shape[0]):
        i=0
        Y1 = 0
        Y2 = 0
        res[tr][2][1] = 0
        for v in range(tris.shape[1]):
            p1 = tris[tr][v]
            p2 = tris[tr][(v+1)%(tris.shape[1])]
            x1 = tris[tr][v][0]
            x2 = p2[0]
            y1 = tris[tr][v][1]
            y2 = p2[1]
            
            badxd = abs(x1-x2)<.001
            goodxd = 1-badxd
            t = (X-x2)/(x1-x2+badxd)
            res[tr][2][1] = 0+badxd
            goodt = 0<=t<=1
            flag = int(goodt)*int(goodxd)

            res[tr][i][0] = (t*(y1-y2)+y2)*flag
            res[tr][i][1] = (t*(p1[2]-p2[2])+p2[2])*flag
            i+=1*flag
        up = int(res[tr][0][0]<res[tr][1][0])
        down = 1-up
        start = res[tr][0]*up + res[tr][1]*down
        end = res[tr][1]*up + res[tr][0]*down
        
        Dstep = (start[1]-end[1])/(start[0]-end[0])
        
        res[tr][0][0] = min(max(start[0],0),h)
        res[tr][0][1] = start[1]+Dstep*(0-start[0])*int(start[0]<0)
        res[tr][1][0] = max(min(end[0],h),0)
        res[tr][1][1] = end[1]
        
        res[tr][2][0] = Dstep
        
            

@guvectorize([(float64[:],float64[:],float64[:,:,:],uint8[:,:])],'(),(h),(a,t,d)->(h,d)',target = 'cpu')
def usetops(X,Y,tops,res):
    
    res *= uint8(0)
    Y +=1000000
    c=0
    dt = 250/tops.shape[0]
    for t in range(tops.shape[0]):
        D=tops[t][0][1]
        
        for yi in range(tops[t][0][0],tops[t][1][0]):
        
            D+= tops[t][2][0]
            
            if D<Y[yi] and D>0:
                #xx= x/250
                #tc = int(((xx%1)**2+((-xx)%1)**2)*500-250)
                #tc = x%250 - ((-x)%125)*((x%))
                res[yi][0] = 100-(t%25)*5
                res[yi][2] = (225 - t*5)
                res[yi][1] = (t)*25
            
                Y[yi] = D

        # res[int(start[0])]=[0,255,255]
        # res[int(end[0])]=[255,255,0]
        c+=1
    # for t in range(4,tops.shape[0]):
        
    #     res[int(tops[t][0][0])]=250
    #     res[int(tops[t][0][0])-1]=250
    #     res[int(tops[t][0][0])-2]=250
    #     res[int(tops[t][0][0])-3]=250
    #     res[int(tops[t][0][0])-4]=250
        #res[int(tops[t][0][0])][2]=250


icoors = coors.astype(np.int64)
coorsy = np.arange(0,height,1,dtype = np.float64)
icoorsy = np.arange(0,height,1,dtype = np.int64)

##################################################polygons end
pygame.font.init()
myfont = pygame.font.SysFont('Comic Sans MS', 30)


t=0
o.l = -supersphere(o.ss)*3

global verts
verts = np.zeros((len(circles),circles[0].shape[0],3),dtype = np.float64)

mousedown=False
lookangles = vec3(0,0,0)

screenarray = np.zeros((width,height,3),dtype=np.uint8)

delta5 = 0

global statevars
statevars = [True,True,True,True] # shading, ui, lines, buttons

def changestate(state):
    s = state
    global statevars
    if s == 0:
        statevars[0]=not statevars[0]
    if s == 1:
        statevars[1]=not statevars[1]
    if s == 2:
        statevars[2]=not statevars[2]
    if s == 3:
        statevars[3]= not statevars[3]

def change_shape():
    global shape_number, verts
    shape_number=(shape_number+1)%len(shapefunclist)
    shapefunclist[shape_number]()
    
    verts = np.zeros((len(circles),circles[0].shape[0],3),dtype = np.float64)

    

import button

button.buts = button.buttons([
    button.button( width-200,200,'Toggle Shading',changestate,(0)),
    button.button( width -200,280,'Toggle UI',changestate,(1)),
    button.button( 10,360,'Toggle Lines',changestate,(2)),
    button.button( 10,440,'Next Shape',change_shape),
    button.button( 10,560,'Toggle buttons',changestate,(3)),
    
    ])




while not done:
    #screenarray = run_shader()
    
    if statevars[0]:
        intar = run_shader().astype(np.uint8)
        #print(intar.shape)
        pygame.surfarray.blit_array(screen,intar)
    else:
        screen.fill((20,20,20))
        
    if statevars[1]:
        ui()

    if statevars[2]:
        show()
    
    if statevars[3]:
        button.buts.update(screen)
    
    tex(shapefunclistnames[shape_number],400)
    
    pygame.display.flip()
    #screen.fill((20,20,20))

    t+=1
    delta5 += clock.tick(60)
    if t%10 ==0:
        
        #print(10/delta5*1000)
        delta5 = 0
    for ev in pygame.event.get():
        if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    done=True
                if ev.key == pygame.K_b:
                    changestate(3)
        if ev.type == pygame.MOUSEBUTTONDOWN:
            button.buts.check_pressed()
            mousedown=True
        if ev.type == pygame.MOUSEBUTTONUP:
            mousedown=False

    if not mousedown:
        m = pygame.mouse.get_rel()
        x = m[0]*math.pi*2/width
        y = m[1]*math.pi*2/height
        lookangles = lookangles+vec3(0,x,y)
        o.ss = lookangles
        o.l = -supersphere(o.ss)*3
    else:
        m = pygame.mouse.get_rel()
        circles=rotcub(circles,m[0])
        circles=rotcub(circles,m[1],1,2)
        
    # delta/=1000
    # lookangles = lookangles+vec3(delta*2,delta/7,delta)
    # o.ss = lookangles
    # o.l = -supersphere(o.ss)*3
    # circles=rotcub(circles,delta*math.pi)
    # circles=rotcub(circles,delta*math.e,1,2)
    
    #t+=1