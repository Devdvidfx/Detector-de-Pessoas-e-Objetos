# Função principal
import cv2

from src.detector import detect_objects, draw_boxes


def main():
    cap = cv2.VideoCapture(0)  # Usar a webcam (ou altere o índice para o ID correto da câmera)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detectar objetos no frame
        detections = detect_objects(frame)

        # Desenhar as caixas e contar o número de pessoas
        person_count = draw_boxes(frame, detections)

        # Exibir o número de pessoas no frame
        cv2.putText(frame, f"Pessoas detectadas: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Mostrar o frame com as detecções
        cv2.imshow("Projeto da equipe 03", frame)

        # Pressione 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()