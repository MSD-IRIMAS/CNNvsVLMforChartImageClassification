import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn import metrics
import pandas as pd


# Custom display of the confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True classes')
    plt.xlabel('Predicted classes')
    plt.tight_layout()


# Calculation of the contingency matrix (or confusion matrix)
def matrice_confusion(y_true, y_predict, classes_name, prompt_type, model_name):
    print('Confusion matrix computing !')
    conf_matrix = metrics.confusion_matrix(y_true, y_predict, labels=classes_name)
    np.set_printoptions(precision=2)
    plt.figure(figsize=(15,15))
    plot_confusion_matrix(conf_matrix, classes=classes_name, title=model_name+" confusion matrix")
    plt.savefig("review/"+model_name+"/Confusion_matrix_prompt_" + prompt_type + ".png")


def get_all_results(Y_true, Y_pred, classes_name, prompt_type, model_name, reviewFilePath):
    accuracy = metrics.accuracy_score(Y_true, Y_pred)
    precision = metrics.precision_score(Y_true, Y_pred, average="macro", zero_division=0.0)
    recall = metrics.recall_score(Y_true, Y_pred, average="macro", zero_division=0.0)
    f1 = metrics.f1_score(Y_true, Y_pred, average="macro", zero_division=0.0)
    # Generate the classification report
    report = metrics.classification_report(Y_true, Y_pred, labels=classes_name,
                                           target_names=classes_name, zero_division=0.0)

    try:
        with open(reviewFilePath, 'a') as review:
            review.write('---------- Global metrics ---------\n')
            review.write(f'Accuracy: {accuracy:.4f} \n')
            review.write('\n')
            review.write(f'Precision: {precision:.4f} \n')
            review.write('\n')
            review.write(f'Recall: {recall:.4f} \n')
            review.write('\n')
            review.write(f'F1-score: {f1:.4f} \n')
            review.write('\n')
            review.write('------ classification report ------\n')
            review.write('\n')
            review.write(str(report) + '\n')
            review.write('\n')
            review.write('************************* END *************************\n')
            review.close()
    finally:
        review.close()
    
    report_dict = metrics.classification_report(Y_true, Y_pred, labels=classes_name,
                                                target_names=classes_name, output_dict=True, zero_division=0.0)
    # Convert the report to DataFrame for easier plotting
    df = pd.DataFrame(report_dict).transpose()
    reportPath = 'review/'+model_name+'/Classif_report_prompt_' + prompt_type + '.csv'
    df.to_csv(reportPath, sep='\t', encoding='utf-8')

    cl_metrics = df[['precision', 'recall', 'f1-score']].iloc[:-2]
    plt.style.use("tableau-colorblind10")
    # Creation of the paired bar chart
    cl_metrics.plot(kind='bar', figsize=(26, 18))
    plt.title('Metrics per class')
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.legend(loc='upper right')
    plt.grid(axis='y')
    plt.savefig('review/'+model_name+'/metrics_per_class_prompt_'+ prompt_type + '.png')

    matrice_confusion(Y_true, Y_pred, classes_name, prompt_type, model_name)


def Y_array_to_class_label_array(Y_array):
    class_label_array = []
    for i in range(len(Y_array)):
        if Y_array[i] == 0:
            class_label_array.append('area')
        elif Y_array[i] == 1:
            class_label_array.append('bar')
        elif Y_array[i] == 2:
            class_label_array.append('barcode')
        elif Y_array[i] == 3:
            class_label_array.append('boxplot')
        elif Y_array[i] == 4:
            class_label_array.append('bubble')
        elif Y_array[i] == 5:
            class_label_array.append('column')
        elif Y_array[i] == 6:
            class_label_array.append('diverging bar')
        elif Y_array[i] == 7:
            class_label_array.append('diverging stacked bar')
        elif Y_array[i] == 8:
            class_label_array.append('donut')
        elif Y_array[i] == 9:
            class_label_array.append('dot strip plot')
        elif Y_array[i] == 10:
            class_label_array.append('heatmap')
        elif Y_array[i] == 11:
            class_label_array.append('line')
        elif Y_array[i] == 12:
            class_label_array.append('line column')
        elif Y_array[i] == 13:
            class_label_array.append('lollipop')
        elif Y_array[i] == 14:
            class_label_array.append('ordered bar')
        elif Y_array[i] == 15:
            class_label_array.append('ordered column')
        elif Y_array[i] == 16:
            class_label_array.append('paired bar')
        elif Y_array[i] == 17:
            class_label_array.append('paired column')
        elif Y_array[i] == 18:
            class_label_array.append('pie')
        elif Y_array[i] == 19:
            class_label_array.append('population pyramid')
        elif Y_array[i] == 20:
            class_label_array.append('proportional stacked bar')
        elif Y_array[i] == 21:
            class_label_array.append('scatter')
        elif Y_array[i] == 22:
            class_label_array.append('spine')
        elif Y_array[i] == 23:
            class_label_array.append('stacked column')
        else:
            class_label_array.append('violin plot')
    
    return class_label_array

