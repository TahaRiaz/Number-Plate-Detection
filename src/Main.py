

import cv2
import numpy as np
import os

import DetectChars
import DetectPlates
import PossiblePlate
from firebase import firebase
from datetime import datetime
import time

# module level variables ##########################################################################
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False


def main():

    ##########FireBase########################################

    
    url = 'https://toll-bbd9b.firebaseio.com'
    fb = firebase.FirebaseApplication(url,None)
    date = str(datetime.now())
    

    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()         

    if blnKNNTrainingSuccessful == False:                               
        print("\nerror: KNN traning was not successful\n")  
        return                                                          
    
    frameadd = 0
    
    while(True):
        
        frameaddx = str(frameadd)
        imgOriginalScene  = cv2.imread("LicPlateImages/"+frameaddx+".jpg") 

        if imgOriginalScene is None:                            
            print("\nerror: image not read from file \n\n")  
            os.system("pause")                                  
            return                                              
        # end if

        listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)          

        listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)        

        #cv2.imshow("imgOriginalScene", imgOriginalScene)            

        if len(listOfPossiblePlates) == 0:                          
            print("\nno license plates were detected\n")  
        else:                                                       
                    

                    
            listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

                    
            licPlate = listOfPossiblePlates[0]

            #cv2.imshow("imgPlate", licPlate.imgPlate)           
            #cv2.imshow("imgThresh", licPlate.imgThresh)

            if len(licPlate.strChars) == 0:                     
                print("\nno characters were detected\n\n")  
                return                                          
            # end if

            drawRedRectangleAroundPlate(imgOriginalScene, licPlate)             

            print("\nlicense plate read from image = " + licPlate.strChars + "\n")  

            ''''name = "UserName"
            url = 'https://toll-bbd9b.firebaseio.com'
            fb = firebase.FirebaseApplication(url,None)
            date = str(datetime.now())'''

            getlicense = fb.get('License','')
            
            key = getlicense.keys()

            
            values = getlicense.values()
            
            
            licPlate.strChar = "AYK234"
            x =licPlate.strChar in key

            aa = getlicense.get(licPlate.strChar,getlicense.values)
            
            
            
            
            officer = "15k-2231"

            getresult = fb.get('Users',officer+"/Type")
            
            
            
              
           

            if(True == x):
                getbalance = int(fb.get('Users',aa+"/Balance"))
             
                if(getbalance >= 100):

                    getbalance =str(getbalance - 100)
                    
                    resulttransaction = fb.put('/Users/'+aa,"Balance",getbalance)
                else:
                    status = "NO MONEY"
                    name = ""+officer
                    resultx = fb.patch('/CarCheck/'+name,{'1':' '+licPlate.strChars+' '+date+' '+status+' /LicPlateImages/'+frameaddx+'.jpg'})
                
                 
                resulttransaction = fb.patch('/Transaction/'+aa,{'1':''+date+' Highway'})
            else:
                
                status = "NO Plate"
                name = ""+officer
                resultx = fb.patch('/CarCheck/'+name,{'1':' '+licPlate.strChars+' '+date+' '+status+' /LicPlateImages/'+frameaddx+'.jpg'})
            

            
            
            #getresult = fb.get('Users',officer+"/Type")

            #if(getbalance == 0):
                    
                    #status = "NO MONEY"
            ##elif(licPlate.strChars != getlicense.key):
                    #status = "Number Plate Not Found"
            
            if(getresult == "Officer"):
                
                aa = getlicense.get(licPlate.strChars,getlicense.values)

                if(licPlate.strChars != getlicense.keys):
                    
                    status = "Number Plate Not Found"
                    name = ""+officer
                    result = fb.patch('/CarCheck/'+name,{'1':''+licPlate.strChars+' '+date+' '+status+' /LicPlateImages/'+frameaddx+'.jpg'})
            else:
                print("No Officer Nearby!!")
                
            
            
            ##############################print(licPlate.strChars)###################################

            #writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)           

            #cv2.imshow("imgOriginalScene", imgOriginalScene)                

            cv2.imwrite("imgOriginalScene.png", imgOriginalScene)           
            frameadd +=1
        

    cv2.waitKey(0)					

         
    return


###################################################################################################
def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):

    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)            

    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_GREEN, 2)         
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_GREEN, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_GREEN, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_GREEN, 2)


###################################################################################################
def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0                            
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0                          
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX                      
    fltFontScale = float(plateHeight) / 30.0                    
    intFontThickness = int(round(fltFontScale * 1.5))           

    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale, intFontThickness)        

            
    ( (intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg ) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)              
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)         

    if intPlateCenterY < (sceneHeight * 0.75):                                                  
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))      
    else:                                                                                       
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))      
    

    textSizeWidth, textSizeHeight = textSize                

    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))           
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))          

            
    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace, fltFontScale, SCALAR_YELLOW, intFontThickness)

    time.sleep(10)
###################################################################################################
if __name__ == "__main__":
    main()


















