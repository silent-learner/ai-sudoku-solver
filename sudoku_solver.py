import os

os.system('color')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np 
import matplotlib.pyplot as plt
import cv2 as cv
from keras.models import load_model # type: ignore
import sys

RED = '\033[31m'
GREEN = '\033[32m'
BLUE = '\033[34m'
RESET = '\033[0m'

model = load_model('best_model.h5.keras')

def findcorners(polygon):
    maxi = -1
    br = []
    tr = []
    bl = []
    tl = []
    for p in polygon:
        point = p[0]
        if(point[0] + point[1] > maxi):
            maxi = point[0] + point[1]
            br = point 

    maxi = -1
    for p in polygon:
        point = p[0]
        if(point[0] - point[1] > maxi):
            maxi = point[0] - point[1]
            tr = point 

    mini = 1e9
    for p in polygon:
        point = p[0]
        if(point[0] + point[1] < mini):
            mini = point[0] + point[1]
            tl = point 

    mini = 1e9
    for p in polygon:
        point = p[0]
        if(point[0] - point[1] < mini):
            mini = point[0] - point[1]
            bl = point 

    return tl ,tr , bl , br

def extract_digit(img):
    digit = img.copy()
    digit = cv.GaussianBlur(digit,(11,11),0)
    digit = cv.blur(digit,(5,5))
    digit = cv.adaptiveThreshold(digit,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,15,2)
    # digit = cv.dilate(digit, kernel)
    contours, _= cv.findContours(digit.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours :
        return np.zeros((128,128))
    largestcontour = max(contours, key=cv.contourArea)
    if cv.contourArea(largestcontour) > 210:
        x, y, w, h = cv.boundingRect(largestcontour)
        digit = digit[y:y+h, x:x+w]
        digit = cv.copyMakeBorder(digit,7,7,7,7,0)
        digit = cv.resize(digit, (128, 128))
        return digit
    else:
        return np.zeros((128,128))

def predictNumber(img):
    if img.sum()/255 == 128*128:
        return [0 , 1]
    else:
        sample = np.reshape(img,(1,128,128,1)) / 255.0
        y_pred = model.predict(sample,verbose=0)
        y_hat = np.argmax(y_pred,axis=1)
        if np.max(y_pred,axis=1) > 0.95:
            return [y_hat[0] , np.max(y_pred,axis=1)]
        else:
            return [0 , np.max(y_pred,axis=1)]

def extract_sudoku(sudoku):

    blur = cv.bilateralFilter(sudoku.copy(),9,50,100)
    thresh = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)

    cv.imwrite('./Outputs/Threshold Image.jpg',thresh)

    thresh = cv.bitwise_not(thresh, thresh)  
    kernel = np.ones((3,3),np.uint8)
    thresh = cv.dilate(thresh, kernel)

    cv.imwrite('./Outputs/Dialated Image.jpg',thresh)


    contours, h = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    polygon = contours[0]
    print("----------------Finding Corners of Sudoku---------------------")
    tl , tr, bl , br = findcorners(polygon)

    largestcontour = cv.circle(sudoku,tr,5,(0,0,255),-1)
    largestcontour = cv.circle(largestcontour,tl,5,(0,0,255),-1)
    largestcontour = cv.circle(largestcontour,br,5,(0,0,255),-1)
    largestcontour = cv.circle(largestcontour,bl,5,(0,0,255),-1)

    cv.imwrite('./Outputs/Corners Found.jpg',largestcontour)

    transform_matrix = cv.getPerspectiveTransform(np.array([tl,tr,bl,br],np.float32),np.array([[0,0],[630,0],[0,630],[630,630]],np.float32))
    extracted_sudoku = cv.warpPerspective(largestcontour,transform_matrix,(630,630))

    cv.imwrite('./Outputs/Extracted Sudoku.jpg',extracted_sudoku)

    extracted_sudoku = cv.bilateralFilter(extracted_sudoku,11,20,20)

    digits_images = []
    for i in range(9):
        for j in range(9):
            digit = extracted_sudoku[70*i:70*(i+1),70*j:70*(j+1)]
            digits_images.append(digit[7:64,7:64])

    i = 0
    extracted_digits = []
    for digit in digits_images:
        extracted_digit = extract_digit(digit)
        extracted_digits.append(np.array(extracted_digit))
        i = i + 1
    
    extracted_digits = 255 - np.array(extracted_digits)

    return extracted_digits, tl , tr, bl ,br

def is_valid(board, row, col, num):
    for i in range(9):
        if board[row][i] == num or board[i][col] == num:
            return False
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(3):
        for j in range(3):
            if board[start_row + i][start_col + j] == num:
                return False
    return True

def solve_sudoku(board):
    for row in range(9):
        for col in range(9):
            if board[row][col] == '0':
                for num in map(str, range(1, 10)):
                    if is_valid(board, row, col, num):
                        board[row][col] = num
                        if solve_sudoku(board):
                            return True
                        board[row][col] = '0'
                return False
    return True

def print_board(board):
    for row in board:
        x = " ".join(row)
        print(f"{BLUE}{x}{RESET}")

def app(sudoku_colored):
    
    sudoku = cv.cvtColor(sudoku_colored,cv.COLOR_BGR2GRAY)

    print("-----------Extracting Sudoku------------------------------")
    extracted_digits , tl ,tr ,bl , br = extract_sudoku(sudoku)

    print("---------------Predicting Each Digit---------------------")
    fig , ax = plt.subplots(9,9,figsize=(15,15),sharex=True,sharey=True)
    i = 0
    grid = []
    for img in extracted_digits:
        ax[i//9][i%9].imshow(img,cmap='gray')
        y_hat , acc = predictNumber(img)
        grid.append(str(y_hat))
        ax[i//9][i%9].title.set_text(f'{y_hat} ({acc})')
        i = i + 1
        print(f"{GREEN} {i}th digit predicted!!! {RESET}")
    
    fig.savefig('./Outputs/Predictions.png')

    sudoku_grid = []
    sudoku_grid.append(grid[:9])
    sudoku_grid.append(grid[9:18])
    sudoku_grid.append(grid[18:27])
    sudoku_grid.append(grid[27:36])
    sudoku_grid.append(grid[36:45])
    sudoku_grid.append(grid[45:54])
    sudoku_grid.append(grid[54:63])
    sudoku_grid.append(grid[63:72])
    sudoku_grid.append(grid[72:])

    print("---------------Solving Sudoku----------------------")

    if solve_sudoku(sudoku_grid):
        print("Solved Sudoku:")
        print_board(sudoku_grid)
    else:
        print("No solution exists.")
        exit(0)

    sudoku_digits = np.zeros((630,630,3))

    missing_digits = np.zeros((630,630,3))

    for i in range(81):
        if grid[i] == '0':
            missing_digits = cv.putText(missing_digits,str(sudoku_grid[i//9][i%9]),(10+70*(i%9),50+70*(i//9)),cv.FONT_HERSHEY_COMPLEX,1.75,(255,255,0),3,cv.LINE_AA)
        else:
            sudoku_digits = cv.putText(sudoku_digits,str(sudoku_grid[i//9][i%9]),(10+70*(i%9),50+70*(i//9)),cv.FONT_HERSHEY_COMPLEX,1.75,(0,255,255),3,cv.LINE_AA)


    cv.imwrite('./Outputs/Missing Digits.jpg',missing_digits)
    cv.imwrite('./Outputs/Sudoku Digits.jpg',sudoku_digits)

    print("--------------Overlaying Solution on original sudoku------------------")
    largest_contour = np.array([tl , tr, bl , br],dtype=np.float32)
    overlay_matrix = cv.getPerspectiveTransform(np.array([[0,0],[630,0],[0,630],[630,630]],np.float32), largest_contour)
    overlay_image = cv.warpPerspective(missing_digits,overlay_matrix,(sudoku.shape[1],sudoku.shape[0]))
    overlayed_sudoku = cv.addWeighted(np.asarray(overlay_image,np.uint8),1,np.asarray(sudoku_colored,np.uint8),0.6,1)
    
    cv.imwrite('./Outputs/Solved Sudoku.jpg',overlayed_sudoku)

    return overlayed_sudoku

if __name__ == '__main__':
    
    sudoku_colored = cv.imread(f'{sys.argv[1]}')
    if sudoku_colored is None:
        print(f'{RED}Invalid Image.{RESET}')
        exit(0)

    overlayed_sudoku = app(sudoku_colored)
    

