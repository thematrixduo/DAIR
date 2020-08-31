import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

num_to_gen = 90000
save_path = '/local/sdb/wd263/my_dsprites/raw_occluded_hh/'

def generate_circle(x,y,r):
    return plt.Circle((x,y), r, color= 'white',fill=True)

def generate_ellipse(x,y,w,h,angle):
    return Ellipse((x,y), w, h, angle = angle, color= 'white',fill=True)
    
def generate_triangle(V):
    return plt.Polygon(V, color= 'white', fill=True)

def generate_square(x,y,l,angle):
    return plt.Rectangle((x,y),l,l,angle=angle, color= 'white',fill=True)

def rand_ellipse(size_low=16.0/64,size_high=28.0/64, hw_ratio = 0.6, margin = 0):
    s = np.random.uniform(size_low,size_high)
    w = s
    h = s*hw_ratio
    angle = np.random.uniform(0,180)
    x = np.random.uniform(s+ margin, 1-s-margin)
    y = np.random.uniform(s+ margin, 1-s-margin)
    return x,y,w,h,angle,s

def rand_square(size_low=15.0/64,size_high=21.0/64, margin = 0):

    l = np.random.uniform(size_low,size_high)
    s = np.sqrt(2*l*l)/2
    angle = np.random.uniform(0,90)
    cx = np.random.uniform(s+ margin, 1-s-margin)
    cy = np.random.uniform(s+ margin, 1-s-margin)
    blv_angle = (-135+angle)*np.pi/180.0
    x = cx + s*np.cos(blv_angle)
    y = cy + s*np.sin(blv_angle)
    return x,y,l,angle,s,cx,cy

def rand_triangle(size_low=7.0/64,size_high=20.0/64, margin = 0):
    s = np.random.uniform(size_low,size_high)
    angle = np.random.uniform(0,360)
    cx = np.random.uniform(s+ margin, 1-s-margin)
    cy = np.random.uniform(s+ margin, 1-s-margin)
    V = []
    start_angle = -150
    angle_interval = 120
    for i in range(3):
        v_angle = (start_angle + i * angle_interval + angle) * np.pi / 180
        vx = cx + s*np.cos(v_angle)
        vy = cy + s*np.sin(v_angle) 
        V.append([vx,vy])
    return np.array(V),cx,cy,s,angle

def generate_obj(obj_type = None):
    if obj_type == None:
        obj_type = np.random.randint(0,3)
    if obj_type == 0:
        cx,cy,w,h,angle,s = rand_ellipse()
        obj = generate_ellipse(cx,cy,w,h,angle)
    if obj_type == 1:
        x,y,l,angle,s,cx,cy = rand_square()
        obj = generate_square(x,y,l,angle)
    if obj_type == 2:
        V,cx,cy,s,angle = rand_triangle()
        obj = generate_triangle(V)
    return obj,cx,cy,s, obj_type

def check_dist(center_list,s_list,x,y,s, d = -0.15):

    check_pass = True
    for i in range(len(center_list)):
        center_i = center_list[i]
        x_i = center_i[0]
        y_i = center_i[1]
        x_dist = (x_i-x)**2
        y_dist = (y_i-y)**2
        threshold = (s_list[i]+s)**2 + d
        #print(x_dist,y_dist,s_list[i],s)
        if x_dist + y_dist < threshold:
            check_pass = False
            break

    return check_pass
 
def check_dist_always_true():
    return True

def generate_step(num_obj, num_obj_max):
    if num_obj:
        obj_type_np=-np.ones(num_obj_max)
        redo = True
        while redo:
            redo = False
            
            center_list = []
            s_list = []
            obj_list = []
            obj_type_list = []
            for i in range(num_obj):
                obj,cx,cy,s, obj_type = generate_obj()
                #print(redo)
                if i>0:
                    check_pass = False
                    counter = 0
                    while not check_pass:
                        #check_pass = check_dist(center_list,s_list,cx,cy,s)
                        check_pass = check_dist_always_true()
                        if not check_pass:
                            obj,cx,cy,s, obj_type = generate_obj(obj_type)
                        counter += 1
                        if counter > 100:
                            redo = True
                            break
                if redo:
                    break
                #print(obj_type,cx,cy,s)
                center_list.append((cx,cy))
                s_list.append(s)
                obj_list.append(obj)
                obj_type_list.append(obj_type)

        for i in range(len(obj_type_list)):
            obj_type_np[i] = obj_type_list[i]

        return obj_list, obj_type_np

def generate(num_to_gen, save_path, num_obj_max = 3):
    obj_type_array = []
    for i in range(num_to_gen):

        num_obj = np.random.randint(1,num_obj_max+1)
        obj_list, obj_type_np = generate_step(num_obj, num_obj_max)
        if i%1000 == 0:
            print(i,obj_type_np)
        obj_type_array.append(obj_type_np)
        fig, ax = plt.subplots(figsize=(0.64,0.64))
        for obj in obj_list:            
            ax.add_artist(obj)

        ax.axis('off')
        filename = save_path + 'img_{}.png'
        fig.savefig(filename.format(i),facecolor='black')
        plt.close()
    np.savez_compressed(save_path + 'labels',labels=np.array(obj_type_array))


generate(num_to_gen,save_path)
