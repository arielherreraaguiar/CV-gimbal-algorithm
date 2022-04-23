'''This algorithm identifies and tracks the sky line and ground plane in real time for outdoor images. ''' 

import numpy as np
import cv2 as cv
import sympy as sp


def fit(x, y): 
    """Curve Fitting Straight line. Return the slope of the line a and the y-intercept b"""
    xbar = sum(x) / len(x)
    ybar = sum(y) / len(y)
    n = len(x) # or len(y)
    numer = sum([xi * yi for xi,yi in zip(x, y)]) - n * xbar * ybar
    denum = sum([xi ** 2 for xi in x]) - n * xbar ** 2
    a = numer / denum
    b = ybar - a * xbar
    return a, b


def distance(x, y): 
    """Straight line distance between two points"""
    x1 = x[0]
    x2 = x[-1]
    y1 = y[0]
    y2 = y[-1]
    d = np.sqrt(((x2-x1) ** 2) + ((y2-y1) ** 2))
    return d


def boundary_removal(img): 
    """Remove edges from the boundary of the image frame."""
    for i in range (1,13):
        img[:,-i] = img[:,-15]
    for i in range (0,12):
        img[:,i] = img[:,15]
    for i in range (0,12):
        img[i,:] = img[15,:]
    return img


def estimate_plane(a, b, c):
    """Estimate the parameters of the plane passing by three points.
    Return:center(float): The center point of the three input points.
    normal(float): The normal to the plane."""
    center = (a + b + c) / 3
    normal = np.cross(b - a, c - a)
    assert(np.isclose(np.dot(b - a, normal), np.dot(c - a, normal)))
    return center, normal


def plane_area(plane_coordinates):
    """Estimate the area of the plane given its coordinates (array)."""
    a = 0
    ox, oy = plane_coordinates[0]
    for x, y in plane_coordinates[1:]:
        a += (x * oy - y * ox)
        ox, oy = x, y
        a_plane = a / 2
    return a_plane


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    vec1: A 3d "source" vector
    vec2: A 3d "destination" vector
    Return: A transform matrix (3x3)
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


#Raspberry Pi camera parameters
dispW = 960
dispH = 540
flip = 2
camSet = 'nvarguscamerasrc !  video/x-raw(memory:NVMM), width = 3264, height = 2464, format = NV12, framerate = 21/1 ! nvvidconv flip-method = '+str(flip)+' ! video/x-raw, width ='+str(dispW)+', height ='+str(dispH)+', format = BGRx ! videoconvert ! video/x-raw, format = BGR ! appsink'
cam = cv.VideoCapture(camSet)

#Uncomment the next line for USB camera
#cam=cv.VideoCapture(0) 

#Perception Variables
last_last_straightline_params = None
last_straightline_params = None
image_counter = 0

while True:
    #Run camera in real time
    ret, output = cam.read()
    cv.imshow('Cam', output)
    if cv.waitKey(1) == ord('q'):
        break

    image_counter +=1
    if image_counter == 1:
        img = output
        img_height = img.shape[0] 
        img_width = img.shape[1] 
        #HFOV = Horizontal Field of View of the camera
        #HFOV_GoPro = 170° ; Raspberry Camera = 62.2°
        HFOV = 62.2 #degrees
        focal_length_pixel = (img_width * 0.5) / np.tan(HFOV * 0.5 * np.pi / 180) #pixels
        #Focal length milimiters 1.7 mm lens GoPro. Raspberry 3.04 mm
        focal_length_mm = 3.04 #milimeters
        img = boundary_removal(img)
        #(x_position, y_position) = Sky line coordinates
        x_position = []
        y_position = []
        for i in range (0, len(img[0, :]), 5):
            for j in range (0, len(img[:, 0]), 5):
                if img[j, i] == 0:
                    x_position.append(i)
                    y_position.append(j)
                    break
        #a = Slope
        #b = y-intercept
        a, b = fit(x_position, y_position)
        angle_a = np.arctan(a) #radians
        last_last_angle = angle_a #radians
        last_last_b = b #pixels

        #Reference Sky line Values
        img_reference = img
        reference_angle = angle_a #radians
        reference_b = b #pixels

        #Reference Sky line Distance
        d_ref = distance(x_position, y_position) #pixels

        #Reference Area below the sky line
        x_line = np.arange(x_position[0], x_position[-1], 1)
        y_line = a * x_line + b
        area_integral = sp.integrate(y_line, (x_line, x_position[0], x_position[-1])) #pixels^2
        area_total = img_height * (x_position[-1] - x_position[0]) #pixels^2
        area_ref = area_total - area_integral #pixels^2

        #Plane 1 Reference Values
        # (p,q) = Center Straight Line Coordinates
        p = x_position[int(len(x_position) / 2)] #pixels
        q = y_position[int(len(y_position) / 2)] #pixels
        #Threshold values
        t_x = 200 #pixels
        t_y = 100 #pixels
        altitude = 300 #milimeters
        # m, n, l = Coordinates of three points in a reference plane
        if p > t_x and q > t_y:
            m = np.array([p, q + 50, altitude * focal_length_pixel / focal_length_mm]) #pixels
            n = np.array([p + 100, q + 100, altitude * focal_length_pixel / focal_length_mm]) #pixels
            l = np.array([p - 100, img_height, altitude * focal_length_pixel / focal_length_mm]) #pixels
        else:
            m = np.array([p + t_x, q + t_y + 50, altitude * focal_length_pixel / focal_length_mm]) #pixels
            n = np.array([p + t_x + 100, q + t_y + 100, altitude * focal_length_pixel / focal_length_mm]) #pixels
            l = np.array([p + t_x - 100, img_height, altitude * focal_length_pixel / focal_length_mm]) #pixels
        #Estimate reference plane center and reference plane normal vector
        center_ref, vec_ref = estimate_plane(m, n, l)

        
    elif image_counter == 2:
        img = output
        img = boundary_removal(img)
        #(x_position, y_position) = Current sky line image coordinates
        x_position = []
        y_position = []
        for i in range (0, len(img[0, :]), 5):
            for j in range (0, len(img[:, 0]), 5):
                if img[j, i] == 0:
                    x_position.append(i)
                    y_position.append(j)
                    break
        #a = Current Slope
        #b = Current y-intercept
        a, b = fit(x_position, y_position)   
        angle_a = np.arctan(a) #radians
        last_angle = angle_a #radians
        last_b = b #pixels
        
        #Increase of current a and b with respect to the reference image
        theta_inc = last_angle - last_last_angle #radians
        b_inc = last_b - last_last_b #pixels

        #Sky line roll angle compensation movement with respect to the reference image
        angle_current = angle_a #radians
        roll_angle_compensate = angle_current - reference_angle #radians
        
        #Sky line pitch angle compensation movement with respect to the reference image
        b_current  = b #pixels
        b_movement = b_current - reference_b #pixels
        #b measured from the center of the image to the reference image
        b_center_ref = reference_b - img_height / 2 #pixels
        #Angle measured from the center of the image to the reference image
        b_center_ref_angle = np.arctan(b_center_ref / focal_length_pixel) #radians
        #b measured from the center of the image to the current sky line image height
        b_total = b_center_ref + b_movement #pixels
        #Angle measured from the center of the image to the current sky line image height
        b_total_angle = np.arctan(b_total / focal_length_pixel) #radians
        #Pitch angle compensation movement
        pitch_angle_compensate = np.arctan(b_total_angle - b_center_ref_angle) #radians

        #Plane 2 Values
        # (p,q) = Current Center Straight Line Coordinates
        p = x_position[int(len(x_position) / 2)] #pixels
        q = y_position[int(len(y_position) / 2)] #pixels
        #Threshold values
        t_x = 200 #pixels
        t_y = 100 #pixels
        altitude = 300 #milimeters
        # m, n, l = Coordinates of three points in a current plane
        if p > t_x and q > t_y:
            m = np.array([p, q + 50, altitude * focal_length_pixel / focal_length_mm]) #pixels
            n = np.array([p + 100, q + 100, altitude * focal_length_pixel / focal_length_mm]) #pixels
            l = np.array([p - 100, img_height, altitude * focal_length_pixel / focal_length_mm]) #pixels
        else:
            m = np.array([p + t_x, q + t_y + 50, altitude * focal_length_pixel / focal_length_mm]) #pixels
            n = np.array([p + t_x + 100, q + t_y + 100, altitude * focal_length_pixel / focal_length_mm]) #pixels
            l = np.array([p + t_x - 100, img_height, altitude * focal_length_pixel / focal_length_mm]) #pixels
        #Estimate plane center and plane normal vector
        center_current, vec_current = estimate_plane(m, n, l)
        #Rotation Matrix between the current plane and the reference plane
        rotation_matrix = rotation_matrix_from_vectors(vec_current, vec_ref)
        #Plane angles compensation
        theta_x = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2]) #radians
        theta_y = np.arctan2(-rotation_matrix[2,0], np.sqrt((rotation_matrix[2,1]) ** 2 + (rotation_matrix[2,2]) ** 2)) #radians
        theta_z = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0]) #radians
        
        print("Roll and Pitch angles [rad]:", roll_angle_compensate, pitch_angle_compensate,'\n')
        print('Theta x, Theta y, Theta z [rad]: ', theta_x, theta_y, theta_z,'\n')
        


    elif image_counter > 2:
        img = output
        img = boundary_removal(img) 
        #Sky line motion prediction values
        theta_pred = theta_inc + last_angle #radians
        b_pred = last_b + b_inc #pixels
        a_p = np.tan(theta_pred) #Sky line slope predicted
        b_p = b_pred #Sky line y-intercept predicted
        
        last_last_angle = last_angle #radians
        last_last_b = last_b #pixels

        #(x_position_predict,  y_position_predict) = Sky line predicted coordinates
        x_position_predict = [i for i in range(0, len(img[0,:]), 5)]
        y_position_predict = []
        for x in x_position_predict:
            y = a_p * x + b_p
            y_position_predict.append(y)

        #(x_position,  y_position) = Current sky line image coordinates
        x_position = x_position_predict 
        y_position = []
        for i,j in zip(x_position, y_position_predict):
            j=int(j)
            if img[j,i] == 0: 
                while (img[j,i] == 0):              
                    j -= 1 
                y_position.append(j)
            else:
                while (img[j,i] == 127):              
                    j += 1 
                y_position.append(j)

        #a = Current Slope
        #b = Current y-intercept
        a, b = fit(x_position, y_position)
        angle_a = np.arctan(a) #radians
        last_angle = angle_a #radians
        last_b = b #pixels
        
        #Increase of current a and b with respect to the reference image
        theta_inc = last_angle - last_last_angle #radians
        b_inc = last_b - last_last_b #pixels

        #Current Sky line Distance
        d_current = distance(x_position, y_position) #pixels
        #Sky line Distance Check
        if d_current <= (1 / 3) * d_ref:
            continue

        #Sky line roll angle compensation movement with respect to the reference image
        angle_current = angle_a #radians
        roll_angle_compensate = angle_current - reference_angle #radians

        #Sky line pitch angle compensation movement with respect to the reference image
        b_current = b #pixels
        b_movement = b_current - reference_b #pixels
        #b measured from the center of the image to the reference image
        b_center_ref = reference_b - img_height / 2 #pixels
        #Angle measured from the center of the image to the reference image
        b_center_ref_angle = np.arctan(b_center_ref / focal_length_pixel) #radians
        #b measured from the center of the image to the current sky line image height
        b_total = b_center_ref + b_movement #pixels
        #Angle measured from the center of the image to the current sky line image height
        b_total_angle = np.arctan(b_total / focal_length_pixel) #radians
        #Pitch angle compensation movement
        pitch_angle_compensate = np.arctan(b_total_angle - b_center_ref_angle) #radians
            
        #Plane 3 Values
        # (p,q) = Current Center Straight Line Coordinates
        p = x_position[int(len(x_position) / 2)] #pixels
        q = y_position[int(len(y_position) / 2)] #pixels
        #Threshold values
        t_x = 200 #pixels
        t_y = 100 #pixels
        altitude = 300 #milimeters
        # m, n, l = Coordinates of three points in a current plane
        if p > t_x and q > t_y:
            m = np.array([p, q + 50, altitude * focal_length_pixel / focal_length_mm]) #pixels
            n = np.array([p + 100, q + 100, altitude * focal_length_pixel / focal_length_mm]) #pixels
            l = np.array([p - 100, img_height, (altitude - 0.4) * focal_length_pixel / focal_length_mm]) #pixels
        else:
            m = np.array([p + t_x, q + t_y + 50, altitude * focal_length_pixel / focal_length_mm]) #pixels
            n = np.array([p + t_x + 100, q + t_y + 100, altitude * focal_length_pixel / focal_length_mm]) #pixels
            l = np.array([p + t_x - 100, img_height, (altitude - 0.4) * focal_length_pixel / focal_length_mm]) #pixels
        
        #Current Area
        plane_coordinates = np.array(m, n, l) #pixels
        area_current = plane_area(plane_coordinates) #pixels^2
        #Plane Area Check
        if area_current <= (1 / 3) * area_ref:
            continue
        #Estimate plane center and plane normal vector
        center_current, vec_current = estimate_plane(m, n, l)
        
        #Rotation Matrix between the current plane and the reference plane
        rotation_matrix = rotation_matrix_from_vectors(vec_current, vec_ref)
        #Plane angles compensation
        theta_x = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2]) #radians
        theta_y = np.arctan2(-rotation_matrix[2,0], np.sqrt((rotation_matrix[2,1]) ** 2 + (rotation_matrix[2,2]) ** 2)) #radians
        theta_z = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0]) #radians
        
        print("Roll and Pitch angles [rad]:", roll_angle_compensate, pitch_angle_compensate,'\n')
        print('Theta x, Theta y, Theta z [rad]: ', theta_x, theta_y, theta_z,'\n')
        
        
cam.release()
cv.destroyAllWindows()
