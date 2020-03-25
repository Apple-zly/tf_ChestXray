
import numpy as np
import pandas as pd
import os

#
def prob2hot(probs):
    preds = probs * 0

    for i in range(probs.shape[0]):
        ind = np.argmax(probs[i])
        preds[i][ind] = 1
    return preds

def get_accuracy(labels, probs):
    return np.sum(labels*prob2hot(probs), axis=1).mean()

# mini-batch数据生成
def generatebatch(X, Y, batch_size, random_state=0):
    n_examples = X.shape[0]
    ind = np.random.permutation(n_examples) # 随机打乱样本顺序
    ind = np.int32(ind)
    for batch_i in range(int(n_examples / batch_size + 0.5)):
        start = batch_i*batch_size
        end = start + batch_size
        batch_xs = X[ind[start:end]]
        batch_ys = Y[ind[start:end]]
        yield batch_xs, batch_ys # 生成每一个batch

# 图像旋转
def img_rotation(batch_imgs, rotate = 0):

    for n in range(rotate):
        N, H, W, C = batch_imgs.shape
        newImg = np.zeros((N, W, H, C), batch_imgs.dtype)
        for i in range(W):
            for j in range(H):
                newImg[:, W - i - 1, j, :] = batch_imgs[:, j, i, :]
        batch_imgs = newImg

    return batch_imgs

# 读取train index 文件，并对类别进行编码
def label_encoder(trainIndex_file, labelEncoded_file, header=None, index_col=None, sep=','):
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder

    trainIndex = pd.read_csv(trainIndex_file, header=header, index_col=index_col, encoding='utf-8', sep=sep)
    LE = LabelEncoder()
    OE = OneHotEncoder()
    LE.fit(trainIndex.iloc[:, 1])

    CLASS = LE.classes_
    CLASS_ID = LE.transform(CLASS)

    targetEncoder = pd.DataFrame(CLASS, columns=['class'], index=CLASS_ID)
    targetEncoder = targetEncoder.sort_index()
    targetEncoder.to_csv(labelEncoded_file, index=None, header=None, encoding='utf-8')

    YE = LE.transform(trainIndex.iloc[:, 1]).reshape([-1, 1])
    Y = OE.fit_transform(np.array(YE).reshape([-1, 1])).todense()
    index = trainIndex.iloc[:, 0]
    return index, Y, CLASS

# 读取 label 文件
def label_read(trainIndex_file, classes_file, header=None, index_col=None, sep=','):

    assert os.path.exists(trainIndex_file), trainIndex_file + "does't exist !\n"
    trainIndex = pd.read_csv(trainIndex_file, header=header, index_col=index_col, encoding='utf-8', sep=sep)

    assert os.path.exists(classes_file), classes_file + "does't exist !\n"
    classes_df = pd.read_csv(classes_file, header=None, encoding='utf-8')

    CLASS = np.array(classes_df[0].values)
    n_class = CLASS.shape[0]
    print('classes:', n_class)

    Y = np.zeros((trainIndex.shape[0], n_class))
    for i in range(n_class):
        lg = np.array(trainIndex.iloc[:, 1] == CLASS[i])
        Y[lg, i] = 1

    index = trainIndex.iloc[:, 0]
    return index, Y, CLASS

# 批量图像读取
def batch_images_read(file_list, newsize=None):
    import cv2
    img = []
    for file in file_list:
        assert os.path.exists(file), file+" does't exist !\n"
        im = cv2.imread(file)

        if newsize:
            im = cv2.resize(im, newsize, interpolation=cv2.INTER_CUBIC)
        im = np.asarray(im, np.float32)
        if len(im.shape)<3:
            im = np.stack((im, im, im), axis=2)
        img.append(im)
    return np.asarray(img)/255.0

def get_f1score(recall, precission):

    return 2*recall*precission/(recall+precission)


def get_Recall_Precision(labels, preds):

    lgtp = (labels == 1) & (preds == 1)
    lgfp = (labels == 0) & (preds == 1)
    lgfn = (labels == 1) & (preds == 0)
    TP = np.sum(lgtp, axis=0).reshape((-1, 1))
    FP = np.sum(lgfp, axis=0).reshape((-1, 1))
    FN = np.sum(lgfn, axis=0).reshape((-1, 1))

    Recall = TP / (TP + FN)

    Precision = TP / (TP + FP)

    return Recall, Precision


def mixMat(labels, preds):

    num_class = labels.shape[1]

    mat = np.zeros((num_class, num_class), np.int64)

    label_ = np.argmax(labels, axis=1)
    pred_ = np.argmax(preds, axis=1)

    for i in range(label_.shape[0]):
        mat[pred_[i], label_[i]] += 1

    return mat

def results_eval(labels, preds, class_names):


    label_cnt = labels.sum(axis=0).reshape((-1, 1))
    pred_cnt = preds.sum(axis=0).reshape((-1, 1))

    lgtp = (labels == 1)&(preds == 1)
    lgfp = (labels == 0)&(preds == 1)
    lgfn = (labels == 1)&(preds == 0)
    TP = np.sum(lgtp, axis=0).reshape((-1, 1))
    FP = np.sum(lgfp, axis=0).reshape((-1, 1))
    FN = np.sum(lgfn, axis=0).reshape((-1, 1))

    Recall = TP / (TP + FN)
    Recall[TP + FN == 0] = -1

    Precision = TP / (TP + FP)
    Precision[TP + FP == 0] = -1

    results = np.hstack((label_cnt, pred_cnt, TP, Recall, Precision))
    index = pd.Index(class_names, name='class')
    columns = ['labels', 'preds', 'TP', 'recall', 'Precision']
    decimals = pd.Series([5, 5], index=columns[-2:])
    results = pd.DataFrame(results, columns=columns, index=index)
    results['labels'] = results['labels'].astype(np.int64)
    results['preds']= results['preds'].astype(np.int64)
    results['TP'] = results['TP'].astype(np.int64)

    accuracy = get_accuracy(labels, preds)

    mat = mixMat(labels, preds)
    mat = pd.DataFrame(mat, index=pd.Index(class_names, name='preds'), columns=class_names)

    print('\n=====================================================================')
    print(results.round(decimals))
    print()
    print(mat)
    print()
    print('\naccuracy:{:3f}\n'.format(accuracy))
    print('=====================================================================\n')

if __name__ == '__main__':

    pass