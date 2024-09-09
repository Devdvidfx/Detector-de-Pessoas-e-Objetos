# Carregar as classes (nomes dos objetos)
import cv2
import numpy as np

from src.config import YOLO_CONFIG_PATH, YOLO_WEIGHTS_PATH, YOLO_CLASSES_PATH, CONFIDENCE_THRESHOLD, NMS_THRESHOLD

with open(YOLO_CLASSES_PATH, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Carregar o modelo YOLOv3
net = cv2.dnn.readNetFromDarknet(YOLO_CONFIG_PATH, YOLO_WEIGHTS_PATH)

# Usar GPU se disponível
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Função para detectar objetos
def detect_objects(frame):
    height, width = frame.shape[:2]

    # Pré-processar o frame (redimensionar e normalizar para 320x320)
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)
    net.setInput(blob)

    # Obter os nomes das camadas de saída do YOLO
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Fazer a detecção de objetos
    detections = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    # Processar as detecções
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > CONFIDENCE_THRESHOLD:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = center_x - w // 2
                y = center_y - h // 2

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Aplicar Non-Maximum Suppression (NMS) para remover caixas redundantes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    # Verificar se existem detecções válidas
    if len(indices) > 0:
        return [(class_ids[i], boxes[i], confidences[i]) for i in indices.flatten()]
    else:
        return []

# Função para desenhar as caixas nas detecções
def draw_boxes(frame, detections):
    person_count = 0
    for (class_id, box, confidence) in detections:
        x, y, w, h = box
        label = str(classes[class_id])

        # Contar apenas pessoas
        if label == "pessoa":
            person_count += 1
            color = (0, 255, 0)  # Verde para pessoas
        else:
            color = (0, 0, 255)  # Vermelho para outros objetos

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = f"{label}: {confidence:.2f}"
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return person_count
