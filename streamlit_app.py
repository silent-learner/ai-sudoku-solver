import os
import streamlit as st
from PIL import Image
import io

os.system('color')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np 
import cv2 as cv
from keras.models import load_model

RED = '\033[31m' 
GREEN = '\033[32m'
BLUE = '\033[34m'
RESET = '\033[0m'

@st.cache_resource
def loadModel():
    model = load_model('best_model.h5.keras')
    return model

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

    thresh = cv.bitwise_not(thresh, thresh)  
    kernel = np.ones((3,3),np.uint8)
    thresh = cv.dilate(thresh, kernel)

    contours, h = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    polygon = contours[0]
    tl , tr, bl , br = findcorners(polygon)

    largestcontour = cv.circle(sudoku,tr,5,(0,0,255),-1)
    largestcontour = cv.circle(largestcontour,tl,5,(0,0,255),-1)
    largestcontour = cv.circle(largestcontour,br,5,(0,0,255),-1)
    largestcontour = cv.circle(largestcontour,bl,5,(0,0,255),-1)


    transform_matrix = cv.getPerspectiveTransform(np.array([tl,tr,bl,br],np.float32),np.array([[0,0],[630,0],[0,630],[630,630]],np.float32))
    extracted_sudoku = cv.warpPerspective(largestcontour,transform_matrix,(630,630))

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
    extracted_digits , tl ,tr ,bl , br = extract_sudoku(sudoku)

    i = 0
    grid = []
    for img in extracted_digits:
        y_hat , acc = predictNumber(img)
        grid.append(str(y_hat))
        i = i + 1
    

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


    if solve_sudoku(sudoku_grid):
        print_board(sudoku_grid)
    else:
        print("Invalid sudoku detected.")
        exit(0)

    sudoku_digits = np.zeros((630,630,3))

    missing_digits = np.zeros((630,630,3))

    for i in range(81):
        if grid[i] == '0':
            missing_digits = cv.putText(missing_digits,str(sudoku_grid[i//9][i%9]),(10+70*(i%9),50+70*(i//9)),cv.QT_FONT_NORMAL,1.75,(250,5,168),3,cv.LINE_AA)
        else:
            sudoku_digits = cv.putText(sudoku_digits,str(sudoku_grid[i//9][i%9]),(10+70*(i%9),50+70*(i//9)),cv.QT_FONT_NORMAL,1.75,(0,255,255),3,cv.LINE_AA)

    largest_contour = np.array([tl , tr, bl , br],dtype=np.float32)
    overlay_matrix = cv.getPerspectiveTransform(np.array([[0,0],[630,0],[0,630],[630,630]],np.float32), largest_contour)
    overlay_image = cv.warpPerspective(missing_digits,overlay_matrix,(sudoku.shape[1],sudoku.shape[0]))
    overlayed_sudoku = cv.addWeighted(np.asarray(overlay_image,np.uint8),1,np.asarray(sudoku_colored,np.uint8),0.5,1)
    
    return overlayed_sudoku

@st.fragment
def download(img_bytes):
    st.download_button(
            label="Download Solved Sudoku",
            data=img_bytes,
            file_name="solved_sudoku.png",
            mime="image/png"
        )

model = loadModel()

st.title("Sudoku Solver")
st.write("Upload an image of an unsolved Sudoku puzzle, and We will solve it for you!")

sudoku_colored = st.file_uploader("Upload Sudoku Image", type=["jpg", "jpeg", "png"])

if sudoku_colored:
    sudoku_colored = Image.open(sudoku_colored)

    is_solved = False

    if st.button("Solve Sudoku"):
        sudoku_colored = np.array(sudoku_colored)
        overlayed_sudoku = app(sudoku_colored)

        overlayed_sudoku = Image.fromarray(overlayed_sudoku)
        st.image(overlayed_sudoku, caption="Solved Sudoku", use_column_width=True)
        is_solved = True

        img_bytes = io.BytesIO()
        overlayed_sudoku.save(img_bytes, format="PNG")
        img_bytes = img_bytes.getvalue()

        download(img_bytes)

    elif not is_solved:
        st.image(sudoku_colored, caption="Unsolved Sudoku", use_column_width=True)

    

