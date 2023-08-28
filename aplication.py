import torch
import cv2
import numpy as np
import time

#largura objetos na vida real. Dados em m.
w_bus = 2.60
w_car = 1.80
w_motorbike = 0.85
w_person = 0.8
w_truck = 2.20
w_van = 2.00

#biblioteca de classes
class_widths = {
    0: w_bus,
    1: w_car,
    2: w_motorbike,
    3: w_person,
    4: w_truck,
    5: w_van
}

#parametros
foco = 634.868 #apos conversão p/pixels
prev_frame_time = 0
d_prev = 0
fps_smooth = 0.0

#lista media movel da vel rel
speed_values = []

#carregar o modelo
model = torch.hub.load('Ultralytics/yolov5', 'custom', 'INSERT PATH TO MODEL HERE', force_reload=True)

#carregar o video
cap = cv2.VideoCapture('INSERT PATH TO VIDEO, OR CAMERA INDEX HERE')

while True:
    
    ret, frame = cap.read()
    rframe = cv2.resize(frame,(640,640))
    
    detect = model(rframe)
    
    new_frame_time = time.time()
    
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    
    fps_smooth = 0.9 * fps_smooth + 0.1 * fps
    
    ####### mascara #######
    mask = np.zeros(rframe.shape[:2], dtype='uint8')
    
    pts = np.array([[0,640],[320,260],[640,640]],np.int32)
    
    cv2.fillPoly(mask,[pts],255)

    masked = cv2.bitwise_and(rframe,rframe,mask = mask)
    #######################
    
    for det in detect.xyxy[0]:
        x1, y1, x2, y2, confidence, classe = [int(coord) for coord in det[:6]]
        
        if cv2.pointPolygonTest(pts, (x1, y1), False) < 0 and cv2.pointPolygonTest(pts, (x2, y2), False) < 0 and cv2.pointPolygonTest(pts, (x1, y2), False) < 0 and cv2.pointPolygonTest(pts, (x2, y1), False) < 0:
            continue
        
        w_im = x2 - x1

        if classe in class_widths:
            class_width = class_widths[classe]
            
            #semelhança de triangulos
            d = (foco * class_width) / w_im
            
            # Calcular velocidade relativa em km/h
            delta_d = d - d_prev
            vel_rel = (delta_d / fps_smooth) * 3.6  # Conversão de m/s para km/h
            
            #adicionar valores a lista da media movel da velocidade
            speed_values.append(vel_rel)
            
            #Mostrar variancia ou desvio padrão
            variancia = np.var(speed_values)
            print("Variância:", variancia)
            #desvio = np.std(speed_values)
            #print("desvio:", desvio)
            
            #calcular a vel rel
            if len(speed_values) >= 2:
                smoothed_speed = sum(speed_values) / len(speed_values)
                cv2.putText(rframe, "{:.2f}".format(smoothed_speed), (x1, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
                speed_values.pop(0)
                
            d_prev = d
    
    #Mostrar os fps na tela
    cv2.putText(rframe, "{:.2f}".format(fps_smooth), (7,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,255,0), 2, cv2.LINE_AA)
    
    cv2.imshow("detection",np.squeeze(detect.render()))
    cv2.imshow("mask", masked)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
