from flask import Flask, render_template, Response
import cv2
import numpy as np
from handTracker import HandTracker
from main import ColorRect
import random

app = Flask(__name__)

# Initialize hand detector and camera
detector = HandTracker(detectionCon=1)
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Define initial canvas
canvas = np.zeros((720, 1280, 3), np.uint8)
color = (255, 0, 0)
brushSize = 5

# Define color rectangles and buttons
colors = [
    ColorRect(300, 0, 100, 100, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))),
    ColorRect(400, 0, 100, 100, (0, 0, 255), "Blue"),
    ColorRect(500, 0, 100, 100, (255, 0, 0), "Red"),
    ColorRect(600, 0, 100, 100, (0, 255, 0), "Green"),
    ColorRect(700, 0, 100, 100, (0, 255, 255), "Yellow"),
    ColorRect(800, 0, 100, 100, (0, 0, 0), "Eraser"),
]

clear = ColorRect(900, 0, 100, 100, (100, 100, 100), "Clear")

pens = [
    ColorRect(1100, 50, 100, 100, (50, 50, 50), "5"),
    ColorRect(1100, 150, 100, 100, (50, 50, 50), "10"),
    ColorRect(1100, 250, 100, 100, (50, 50, 50), "15"),
    ColorRect(1100, 350, 100, 100, (50, 50, 50), "20"),
]

colorsBtn = ColorRect(200, 0, 100, 100, (120, 255, 0), "Colors")
penBtn = ColorRect(1100, 0, 100, 50, color, "Pen")
boardBtn = ColorRect(50, 0, 100, 100, (255, 255, 0), "Board")
whiteBoard = ColorRect(50, 120, 1020, 580, (255, 255, 255), alpha=0.6)

coolingCounter = 20
hideBoard = True
hideColors = True
hidePenSizes = True


@app.route('/')
def index():
    return render_template('index.html')


def generate_frames():
    global canvas, color, brushSize, coolingCounter, hideBoard, hideColors, hidePenSizes

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))
        frame = cv2.flip(frame, 1)

        detector.findHands(frame)
        positions = detector.getPostion(frame, draw=False)
        upFingers = detector.getUpFingers(frame)

        if upFingers:
            x, y = positions[8][0], positions[8][1]
            if upFingers[1] and not whiteBoard.isOver(x, y):
                px, py = 0, 0

                if not hidePenSizes:
                    for pen in pens:
                        if pen.isOver(x, y):
                            brushSize = int(pen.text)
                            pen.alpha = 0
                        else:
                            pen.alpha = 0.5

                if not hideColors:
                    for cb in colors:
                        if cb.isOver(x, y):
                            color = cb.color
                            cb.alpha = 0
                        else:
                            cb.alpha = 0.5

                    if clear.isOver(x, y):
                        clear.alpha = 0
                        canvas = np.zeros((720, 1280, 3), np.uint8)
                    else:
                        clear.alpha = 0.5

                if colorsBtn.isOver(x, y) and not coolingCounter:
                    coolingCounter = 10
                    colorsBtn.alpha = 0
                    hideColors = not hideColors
                    colorsBtn.text = 'Colors' if hideColors else 'Hide'
                else:
                    colorsBtn.alpha = 0.5

                if penBtn.isOver(x, y) and not coolingCounter:
                    coolingCounter = 10
                    penBtn.alpha = 0
                    hidePenSizes = not hidePenSizes
                    penBtn.text = 'Pen' if hidePenSizes else 'Hide'
                else:
                    penBtn.alpha = 0.5

                if boardBtn.isOver(x, y) and not coolingCounter:
                    coolingCounter = 10
                    boardBtn.alpha = 0
                    hideBoard = not hideBoard
                    boardBtn.text = 'Board' if hideBoard else 'Hide'
                else:
                    boardBtn.alpha = 0.5

            elif upFingers[1] and not upFingers[2]:
                if whiteBoard.isOver(x, y) and not hideBoard:
                    cv2.circle(frame, positions[8], brushSize, color, -1)
                    if px == 0 and py == 0:
                        px, py = positions[8]
                    if color == (0, 0, 0):
                        cv2.line(canvas, (px, py), positions[8], color, 20)
                    else:
                        cv2.line(canvas, (px, py), positions[8], color, brushSize)
                    px, py = positions[8]
            else:
                px, py = 0, 0

        colorsBtn.drawRect(frame)
        cv2.rectangle(frame, (colorsBtn.x, colorsBtn.y), (colorsBtn.x + colorsBtn.w, colorsBtn.y + colorsBtn.h),
                      (255, 255, 255), 2)

        boardBtn.drawRect(frame)
        cv2.rectangle(frame, (boardBtn.x, boardBtn.y), (boardBtn.x + boardBtn.w, boardBtn.y + boardBtn.h),
                      (255, 255, 255), 2)

        if not hideBoard:
            whiteBoard.drawRect(frame)
            canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, img_inv = cv2.threshold(canvas_gray, 20, 255, cv2.THRESH_BINARY_INV)
            img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
            frame = cv2.bitwise_and(frame, img_inv)
            frame = cv2.bitwise_or(frame, canvas)

        if not hideColors:
            for c in colors:
                c.drawRect(frame)
                cv2.rectangle(frame, (c.x, c.y), (c.x + c.w, c.y + c.h), (255, 255, 255), 2)

            clear.drawRect(frame)
            cv2.rectangle(frame, (clear.x, clear.y), (clear.x + clear.w, clear.y + clear.h), (255, 255, 255), 2)

        penBtn.color = color
        penBtn.drawRect(frame)
        cv2.rectangle(frame, (penBtn.x, penBtn.y), (penBtn.x + penBtn.w, penBtn.y + penBtn.h), (255, 255, 255), 2)
        if not hidePenSizes:
            for pen in pens:
                pen.drawRect(frame)
                cv2.rectangle(frame, (pen.x, pen.y), (pen.x + pen.w, pen.y + pen.h), (255, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
