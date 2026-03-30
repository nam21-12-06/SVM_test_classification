from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

def evaluate(y_true, y_pred, target_names, show_cm = False,save_cm= False):
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names= target_names)

    print("Accuracy: ", accuracy)
    print("\nClassification report:\n")
    print(report)

    #Confusion matrix
    if save_cm:
        os.makedirs("outputs", exist_ok=True)

        plt.figure(figsize=(10, 8))
        plt.imshow(cm)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()

        plt.savefig("outputs/confusion_matrix.png")
        print("Confusion matrix saved to outputs/confusion_matrix.png")

    if show_cm:
        plt.show()

    plt.close()