import cv2
from ultralytics import YOLO

# Charger modèle
model = YOLO(r"C:/Users/MSI/OneDrive/Desktop/version 2 stage/scripts/runs/detect/train3/weights/best.pt")

# Charger l'image
img_path = r"C:/Users/MSI/OneDrive/Desktop/version 2 stage/DATA/test/images/281_jpg.rf.12b66178c1d2f6991de7ae1e75ab669a.jpg"
img = cv2.imread(img_path)

if img is None:
    print("Erreur : impossible de charger l'image.")
    exit()

# Convertir en RGB (YOLOv8 s'attend souvent à RGB)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Faire la prédiction
results = model(img_rgb)

# Image annotée (en RGB)
annotated_img_rgb = results[0].plot()

# Convertir en BGR pour OpenCV
annotated_img = cv2.cvtColor(annotated_img_rgb, cv2.COLOR_RGB2BGR)

# Redimensionner l'image pour qu'elle tienne dans une fenêtre de 800x600 max
max_width, max_height = 800, 600
height, width = annotated_img.shape[:2]
scaling_factor = min(max_width / width, max_height / height, 1)  # jamais agrandir

if scaling_factor < 1:
    annotated_img = cv2.resize(annotated_img, (int(width * scaling_factor), int(height * scaling_factor)))

# Créer une fenêtre redimensionnable
cv2.namedWindow("Détection Car Logo", cv2.WINDOW_NORMAL)

# Afficher l'image annotée
cv2.imshow("Détection Car Logo", annotated_img)

print("Appuyez sur 'q' ou ESC pour fermer la fenêtre.")

# Attendre la touche 'q' ou ESC
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cv2.destroyAllWindows()
