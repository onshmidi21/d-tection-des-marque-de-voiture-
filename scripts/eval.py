from ultralytics import YOLO

def main():
    model = YOLO(r"C:/Users/MSI/OneDrive/Desktop/version 2 stage/scripts/runs/detect/train3/weights/best.pt")

    metrics = model.val(
        data=r"C:/Users/MSI/OneDrive/Desktop/version 2 stage/DATA/data.yaml",
        workers=0
    )

    print("\n📊 Résultats principaux de l'évaluation :")
    print(f"🎯 mAP@0.5:0.95 = {metrics.box.map:.4f}")
    print(f"🎯 mAP@0.5     = {metrics.box.map50:.4f}")
    print(f"📈 Précision   = {metrics.box.mp:.4f}")
    print(f"🔍 Rappel      = {metrics.box.mr:.4f}")

    # ➕ Afficher les résultats par classe
    print("\n📌 Résultats détaillés par classe :")
    for i, name in enumerate(metrics.names.values()):
        p, r, ap50, ap = metrics.box.class_result(i)
        print(f"{name:12} | P={p:.3f} | R={r:.3f} | AP50={ap50:.3f} | AP={ap:.3f}")

if __name__ == "__main__":
    main()
