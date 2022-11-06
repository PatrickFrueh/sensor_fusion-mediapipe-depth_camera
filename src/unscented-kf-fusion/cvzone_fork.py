"""
Hand Tracking Module
By: Computer Vision Zone
Website: https://www.computervision.zone/
"""

import cv2
import mediapipe as mp
import math


class HandDetector:
    """
    Finds Hands using the mediapipe library. Exports the landmarks
    in pixel format. Adds extra functionalities like finding how
    many fingers are up or the distance between two fingers. Also
    provides bounding box info of the hand found.
    """

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, minTrackCon=0.5):
        """
        :param mode: In static mode, detection is done on each image: slower
        :param maxHands: Maximum number of hands to detect
        :param detectionCon: Minimum Detection Confidence Threshold
        :param minTrackCon: Minimum Tracking Confidence Threshold
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.minTrackCon,
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []

    def findHands(
        self, img, boxW_fix=0, boxH_fix=0, draw=True, flipType=True, fix_box=False
    ):
        """
        Finds hands in a BGR image.
        :param img: Image to find the hands in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(
                self.results.multi_handedness, self.results.multi_hand_landmarks
            ):
                myHand = {}
                ## lmList
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py = int(lm.x * w), int(lm.y * h)
                    mylmList.append([px, py])
                    xList.append(px)
                    yList.append(py)

                ## bbox
                if fix_box == False:
                    xmin, xmax = min(xList), max(xList)
                    ymin, ymax = min(yList), max(yList)
                    boxW, boxH = xmax - xmin, ymax - ymin
                    bbox = xmin, ymin, boxW, boxH
                    cx, cy = bbox[0] + (bbox[2] // 2), bbox[1] + (bbox[3] // 2)

                    myHand["lmList"] = mylmList
                    myHand["bbox"] = bbox
                    myHand["center"] = (cx, cy)

                    if flipType:
                        if handType.classification[0].label == "Right":
                            myHand["type"] = "Left"
                        else:
                            myHand["type"] = "Right"
                    else:
                        myHand["type"] = handType.classification[0].label
                    allHands.append(myHand)

                    ## draw
                    if draw:
                        self.mpDraw.draw_landmarks(
                            img, handLms, self.mpHands.HAND_CONNECTIONS
                        )
                        cv2.rectangle(
                            img,
                            (bbox[0] - 20, bbox[1] - 20),
                            (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                            (255, 0, 255),
                            2,
                        )
                        cv2.putText(
                            img,
                            myHand["type"],
                            (bbox[0] - 30, bbox[1] - 30),
                            cv2.FONT_HERSHEY_PLAIN,
                            2,
                            (255, 0, 255),
                            2,
                        )

                # bbox_fixed
                if fix_box == True:
                    xmin, xmax = min(xList), max(xList)
                    ymin, ymax = min(yList), max(yList)
                    # boxW, boxH = xmax - xmin, ymax - ymin
                    boxW, boxH = xmin - 20, ymax + 20
                    # fix width, height
                    bbox = boxW, boxH, boxW_fix, boxH_fix
                    cx, cy = bbox[0] + (bbox[2] // 2), bbox[1] + (bbox[3] // 2)

                    myHand["lmList"] = mylmList
                    myHand["bbox"] = bbox
                    myHand["center"] = (cx, cy)

                    if flipType:
                        if handType.classification[0].label == "Right":
                            myHand["type"] = "Left"
                        else:
                            myHand["type"] = "Right"
                    else:
                        myHand["type"] = handType.classification[0].label
                    allHands.append(myHand)

                    # draw
                    if draw:

                        # self.mpDraw.draw_landmarks(img, handLms,
                        #                            self.mpHands.HAND_CONNECTIONS)
                        cv2.rectangle(
                            img,
                            (bbox[0], bbox[1]),
                            (bbox[0] + bbox[2], bbox[1] - bbox[3]),
                            (255, 255, 0),
                            2,
                        )
                        cv2.putText(
                            img,
                            myHand["type"],
                            (bbox[0] - 30, bbox[1] - 30),
                            cv2.FONT_HERSHEY_PLAIN,
                            2,
                            (255, 255, 0),
                            2,
                        )
        if draw:
            return allHands, img  # return rectangle
        else:
            return allHands

    def fingersUp(self, myHand):
        """
        Finds how many fingers are open and returns in a list.
        Considers left and right hands separately
        :return: List of which fingers are up
        """
        myHandType = myHand["type"]
        myLmList = myHand["lmList"]
        if self.results.multi_hand_landmarks:
            fingers = []
            # Thumb
            if myHandType == "Right":
                if myLmList[self.tipIds[0]][0] > myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if myLmList[self.tipIds[0]][0] < myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # 4 Fingers
            for id in range(1, 5):
                if myLmList[self.tipIds[id]][1] < myLmList[self.tipIds[id] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img=None):
        """
        Find the distance between two landmarks based on their
        index numbers.
        :param p1: Point1
        :param p2: Point2
        :param img: Image to draw on.
        :param draw: Flag to draw the output on the image.
        :return: Distance between the points
                 Image with output drawn
                 Line information
        """

        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            return length, info, img
        else:
            return length, info

    def predictLandmark(self, dist_init_plus1, dist_current, p1, p2, img, draw=True):
        """
        predict plausible landmark+1 based on trustworthy landmarks (p1,p2)
        :param p1: Point1
        :param p2: Point2
        :param dist_init_plus1: initiated distance - p2 to p3
        :param dist_current: current distance - p1 to p2
        :param img: input image
        :return: corrected
                 landmark
        """
        x1, y1 = p1
        x2, y2 = p2
        corrected_distance = dist_current + dist_init_plus1
        current_distance = dist_current

        # ratio of distances: for t>1 - point outside line between x1,y1 and x2,y2
        t = corrected_distance / current_distance
        x_c, y_c = ((1 - t) * x1 + t * x2), ((1 - t) * y1 + t * y2)
        if draw:
            cv2.circle(
                img, (int(x_c), int(y_c)), radius=6, color=(255, 255, 0), thickness=-1
            )

        landmark_plus1 = x_c, y_c

        return landmark_plus1


def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(detectionCon=0.8, maxHands=2)
    while True:
        # Get image frame
        success, img = cap.read()
        # Find the hand and its landmarks
        hands, img = detector.findHands(img)  # with draw
        # hands = detector.findHands(img, draw=False)  # without draw

        if hands:
            # Hand 1
            hand1 = hands[0]
            lmList1 = hand1["lmList"]  # List of 21 Landmark points
            bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
            centerPoint1 = hand1["center"]  # center of the hand cx,cy
            handType1 = hand1["type"]  # Handtype Left or Right

            fingers1 = detector.fingersUp(hand1)

            if len(hands) == 2:
                # Hand 2
                hand2 = hands[1]
                lmList2 = hand2["lmList"]  # List of 21 Landmark points
                bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
                centerPoint2 = hand2["center"]  # center of the hand cx,cy
                handType2 = hand2["type"]  # Hand Type "Left" or "Right"

                fingers2 = detector.fingersUp(hand2)

                # Find Distance between two Landmarks. Could be same hand or different hands
                length, info, img = detector.findDistance(
                    lmList1[8], lmList2[8], img
                )  # with draw
                # length, info = detector.findDistance(lmList1[8], lmList2[8])  # with draw
        # Display
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
