import cv2
import face_recognition
import pickle
import os

def cargar_base_datos():
    rostros = []
    nombres = []
    for archivo in os.listdir("data"):
        if archivo.endswith(".pkl"):
            with open(f"data/{archivo}", "rb") as f:
                rostros.append(pickle.load(f))
                nombres.append(archivo.replace(".pkl", ""))
    return rostros, nombres

def login():
    rostros_db, nombres_db = cargar_base_datos()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Guía visual
        alto, ancho, _ = frame.shape
        centro = (ancho // 2, alto // 2)
        ejes = (int(ancho * 0.25), int(alto * 0.4))
        
        # Procesar frame (escala 1/4 para velocidad)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb_small)
        encodings = face_recognition.face_encodings(rgb_small, boxes)

        color_oval = (255, 255, 255) # Blanco por defecto
        mensaje = "ESPERANDO ROSTRO..."

        for face_encoding in encodings:
            # Comparar con la base de datos
            matches = face_recognition.compare_faces(rostros_db, face_encoding, tolerance=0.5)
            
            if True in matches:
                idx = matches.index(True)
                nombre_usuario = nombres_db[idx].upper()
                mensaje = f"ACCESO CONCEDIDO: {nombre_usuario}"
                color_oval = (0, 255, 0) # Verde
            else:
                mensaje = "ACCESO DENEGADO"
                color_oval = (0, 0, 255) # Rojo

        # Dibujar Interfaz
        cv2.ellipse(frame, centro, ejes, 0, 0, 360, color_oval, 2)
        cv2.rectangle(frame, (0, 0), (ancho, 40), (0,0,0), -1)
        cv2.putText(frame, mensaje, (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, color_oval, 2)

        cv2.imshow("Login Biometrico", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    login()