import cv2
import numpy as np
import pandas as pd

def get_full_parking_lots(points,car_locations):
    '''
    Returns full parking lots by checking car locations with line points.
    '''

    # We are getting only x coordinates from points dataframe.
    points = points.iloc[2:,:].sort_values(by='x1').reset_index().iloc[:,1:]
    points = points[points.index % 2 == 0]
    x_coordinates = points.x1.values

    # Creating empty array to store full parking lots
    full_parking_lots = []
    
    # Running into x_coordinates and checking all values with all car locations; if there is a car inside of lines, we are appending it to full_parking_lots.
    for i in range(len(x_coordinates)):
        for car in car_locations:
            if x_coordinates[i] <  car[0] and x_coordinates[i+1] > car[0]:
                full_parking_lots.append(i+1)
                
    return full_parking_lots

def get_area(w,h):
    '''
    Returns rectangle area if rectangle's area bigger than 500, otherwise it returns -1
    '''
    return w*h if w*h >500 else -1

# All numbered parking area locations
ALL_PARKING_LOTS = [1,2,3,4,5]

# Creating car classifier.
car_classifier = cv2.CascadeClassifier('haarcascade/haarcascade_car.xml')

# Reading the image which is we are going to work with and displaying it.
image = cv2.imread('data/parking_lot_new.jpg')
cv2.imshow("Original", image)
cv2.waitKey(0)

# Grayscale and Canny Edges extracted
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 70, 200, apertureSize = 3)

# Used car classifier to detect cars in the image
cars = car_classifier.detectMultiScale(image, 1.11,0)

# Displaying edges image
cv2.imshow("edges", edges)
cv2.waitKey(0)

# Getting lines from edges image
lines = cv2.HoughLines(edges, 1, np.pi/90, 200)

# Creating points and car_locations arrays to store line points and car locations
points = np.array([])
car_locations = np.array([])

# Storing line points by extracting them.
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    points = np.append(points,[x1,x2,y1,y2]).reshape(-1,4)

# Creating points_df datafrane to use pandas features.
points_df =pd.DataFrame(points, columns = ['x1','x2','y1','y2'])
points_df.sort_values(by='x1', inplace=True)

# Storing car locations while getting x_center and y_center.
for (x,y,w,h) in cars:
    if(get_area(w,h)) != -1:
        car_locations = np.append(car_locations,[(x+x+w)/2, (y+y+h)/2]).reshape(-1,2)


# Getting full parking lots with our get_full_parking_lots function.
full_parking_lots = get_full_parking_lots(points_df,car_locations)

# Writing parking lots numbers to the image
for p in range(0,len(points_df),2):
    x1,x2,y1,y2 = points_df.iloc[p,:].values
    cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    if p != 0 :
        cv2.putText(image, org = (int((x1+x2) / 2 + 65), int(image.shape[0] / 2)), text = str(int(p/2)), fontFace = cv2.FONT_HERSHEY_SIMPLEX,
               fontScale = 1.0, color = (0,0,255), thickness = 2)

# Creating x_coordinates variable to use it after on writing parking lots' states
points_set = points_df.iloc[2:,:].sort_values(by='x1').reset_index().iloc[:,1:]
points_set = points_set[points_set.index % 2 == 0]
x_coordinates = points_set.x1.values       
        
# Writing "EMPTY" to empty parking lots on image
for i in ALL_PARKING_LOTS:
    if i not in full_parking_lots:
        cv2.putText(image, org = (int(x_coordinates[i]-120),250), text = "Empty", fontFace = cv2.FONT_HERSHEY_SIMPLEX,
               fontScale = 1.0, color = (0,255,255), thickness = 2)        
        
        
# Printing full and empty parking lots to console
print("FULL PARKING LOTS:")
print(set(full_parking_lots))
print()
print("EMPTY PARKING LOTS:")
print(set(ALL_PARKING_LOTS) - set(full_parking_lots))

# Displaying image's final version
cv2.imshow('Final', image)
cv2.waitKey(0)
cv2.destroyAllWindows()