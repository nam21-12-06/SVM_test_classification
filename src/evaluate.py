import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns # Thêm seaborn để vẽ ma trận đẹp hơn

def evaluate(y_true, y_pred, target_names, show_cm=True, save_cm=True):
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=target_names)
    
    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification report:\n")
    print(report)

    if save_cm or show_cm:
        plt.figure(figsize=(12, 10))

        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                    xticklabels=target_names, yticklabels=target_names)
        
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()

        if save_cm:
            if not os.path.exists("outputs"):
                os.makedirs("outputs")
            elif os.path.isfile("outputs"): # Nếu trùng tên với một file
                os.remove("outputs")        # Xóa file đó đi
                os.makedirs("outputs")      # Tạo lại thư mục

            plt.savefig("outputs/confusion_matrix.png")
            print("Confusion matrix is successfully stored in outputs/confusion_matrix.png")

        if show_cm:
            plt.show()

    plt.close()