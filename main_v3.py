import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

from define import *


def extract_feature(data):
    mean_ft = np.mean(data, axis=1)
    std_ft = np.std(data, axis=1)
    max_ft = np.max(data, axis=1)
    min_ft = np.min(data, axis=1)
    var_ft = np.var(data, axis=1)
    med_ft = np.median(data, axis=1)
    sum_ft = np.sum(data, axis=1)
    features = np.array([mean_ft, std_ft, max_ft, min_ft, var_ft, med_ft, sum_ft]).T
    return features


def segment(data, data_index):
    row, col = np.where(data_index >= len(data))
    uniq_row = len(np.unique(row))
    if uniq_row > 0 and row[0] > 0:
        data_seg_ = data[data_index[:-uniq_row, :]]
        data_seg_end = data[data_index[-uniq_row, 0]:]
        data_seg_feat = extract_feature(data_seg_)
        data_seg_end_feat = extract_feature(np.expand_dims(data_seg_end, axis=0))
        data_seg_feat = np.concatenate([data_seg_feat, data_seg_end_feat], axis=0)
    elif uniq_row > 0 and row[0] == 0:
        # data_seg_ = data[data_index[:-uniq_row, :]]
        data_seg_feat = extract_feature(np.expand_dims(data, axis=0))
    else:
        data_seg_ = data[data_index]
        data_seg_feat = extract_feature(data_seg_)

    a=0
    return data_seg_feat

def main():
    # Load data
    df = pd.read_csv(DATA_PATH)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df_label = pd.read_csv(LABEL_PATH)
    df_label = df_label.loc[:, ~df_label.columns.str.contains('^Unnamed')]

    # Filter by user id
    df_label = df_label[df_label['user_id'] == USER_ID]

    # Filter 2-days data Sept 15 and 16
    df_label = df_label[
        (df_label["started_at"] >= '2022-09-15 00:00:00') & (df_label["finished_at"] <= '2022-09-16 23:59:59')]

    # remove activity having too long duration
    activity_name = list(df_label['activity'].unique())
    delete_ind = []
    for name in activity_name:
        each_activity_df = df_label[df_label['activity'] == name]
        duration = pd.to_datetime(each_activity_df['finished_at']) - pd.to_datetime(each_activity_df['started_at'])
        mean_duration = duration.mean()
        delete_ind.extend(list(duration[duration > mean_duration * SCALE_FACTOR].index))

    df_label.drop(delete_ind, axis=0, inplace=True)
    df_label = df_label.reset_index(drop=True)

    # Extract features
    seg_label_list = []
    seg_features_list = []

    for i in range(len(df_label)):
        seg = df[(df["timestamp"] >= df_label['started_at'][i]) & (df["timestamp"] <= df_label['finished_at'][i])]
        seg_label = df_label["activity"][i]
        if len(seg) > 0:
            data_len = len(seg)
            acc_x = np.array(seg['x'])
            acc_y = np.array(seg['y'])
            acc_z = np.array(seg['z'])
            data_index = np.arange(FEATURE_LEN)[None, :] + \
                         np.arange(0, data_len, int(OVERLAP_RATE*FEATURE_LEN))[:, None]
            feat_x = segment(acc_x, data_index)
            feat_y = segment(acc_y, data_index)
            feat_z = segment(acc_z, data_index)
            feat_seg = np.concatenate([feat_x, feat_y, feat_z], axis=1)

            seg_features_list.extend(feat_seg)
            seg_label_list.extend([seg_label]*len(feat_seg))
            a=0

    # Training
    model_ml = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    X_train, X_test, y_train, y_test = train_test_split(seg_features_list, seg_label_list, test_size=0.3,
                                                        random_state=42)

    model_ml.fit(X_train, y_train)
    # Predict
    y_predict = model_ml.predict(X_test)
    y_pred_proba = model_ml.predict_proba(X_test)
    print(classification_report(y_test, y_predict))
    confusion_matrix(y_test, y_predict)

    n_class = len(np.unique(np.array(y_train)))
    label_dict = {'Report creation': 1,
                  'Echocardiography': 2,
                  'electro-cardiogram': 0,
                  'Vascular echo': 3,
                  }
    y_test_convert = [label_dict[i] for i in y_test]
    y_score = label_binarize(y_test_convert, classes=np.arange(n_class))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_class):
        fpr[i], tpr[i], _ = roc_curve(y_score[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_score.ravel(), y_pred_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_class)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_class):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_class

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(n_class), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Some extension of Receiver operating characteristic to multiclass")
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    main()

