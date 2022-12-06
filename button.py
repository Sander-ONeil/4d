
import pygame

pygame.font.init()
global myfont
myfont = pygame.font.SysFont('Sans Sherif', 38)

global Mouse_on_a_button
Mouse_on_a_button = False

def tex(st,w,h,screen=None):
    textsurface = myfont.render(str(st), True, (250, 250, 250))
    
    screen.blit(textsurface,(w,h))

class button:
    def __init__(self,x,y,text,f,args=None):
        self.x = x
        self.y = y
        self.text = str(text)
        self.f = f
        self.text_rect =  myfont.render(self.text, True, (250, 250, 250))
        self.h = self.text_rect.get_height()
        self.w = self.text_rect.get_width()
        self.args = args
    
    def get_rect(self):
        return((self.x,self.y,self.w+10,self.h+10))
    
    def mouse_is_on_button(self):
        x,y,w,h = self.get_rect()

        m = pygame.mouse.get_pos()
        return x<m[0]<x+w and y<m[1]<y+h
    
    def update(self,screen):
        x,y,w,h = self.get_rect()
        
        pygame.draw.rect(screen,(100,100,100),[x-5,y-5,w+10,h+10])
        pygame.draw.rect(screen,(0,0,0),[x,y,w,h])
        
        
        global Mouse_on_a_button
        
        if self.mouse_is_on_button():
            Mouse_on_a_button = True
            pygame.draw.rect(screen,(100,100,100),[x,y,w,h])
        tex(self.text,x+5,y+5,screen)

                            
        
        
        return self.mouse_is_on_button()
    def run(self):
        if self.args is None:
            self.f()
        else:
            self.f(self.args)
    def check_pressed(self):
        
        if self.mouse_is_on_button():
            self.run()
            return True
        return False
            

class buttons:
    def __init__(self,bs):
        self.bs = bs
    def update(self,screen):
        global Mouse_on_a_button
        Mouse_on_a_button = False
        for b in self.bs:
            moab = b.update(screen)
            if moab == True:
                Mouse_on_a_button = True
    def check_pressed(self):
        for b in self.bs:
            if b.check_pressed():
                return True
        return False
            
def BF():
    print('bbbb')

bu = button(100,100,'press',BF)

