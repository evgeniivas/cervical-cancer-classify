import segmentation
import cv2
import mahotas as mh

import numpy as np
import pandas as pd
import math
from sklearn.utils import shuffle
from sklearn.cluster import KMeans

from sklearn.externals import joblib

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss

def chist(img):
    img = img // 64
    r,g,b = img.transpose((2,0,1))
    pixels = 1 * r + 4 * b + 16 * g
    hist = np.bincount(pixels.ravel(),minlength = 64)
    hist = hist.astype(float)
    hist = np.log1p(hist)
    return hist


def make_raw_features(img,img_id):
    w = img.shape[0]
    h = img.shape[1]
    if w>4 and h>4: 
        sift = cv2.xfeatures2d.SIFT_create()
        surf = cv2.xfeatures2d.SURF_create()

        haralick = np.ravel(mh.features.haralick(img))
        color_hist = chist(img)
        pftas=mh.features.pftas(img)
        
        img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
        _ , sift_desc = sift.detectAndCompute(img,None)
        _ , surf_desc = surf.detectAndCompute(img,None)
        lbp = mh.features.lbp(img,radius=8,points=6)

        return [img_id,haralick,color_hist,pftas,sift_desc,surf_desc,lbp]

def fix_desc(desc):
    result = []
    n = len(desc[0][0])
    for i in desc:
        if i is not None:
            result.append(i)
        else:
            result.append([0 for i in xrange(0,n)])        
    return result
    
def concatenate_desc(desc):
    result = np.concatenate(desc)
    return result

def bow(desc,k,name,check=True):
    desc= [desc]
    if check:
        desc = fix_desc(desc)

    km = joblib.load('./models/'+str(name)+'.pkl')
    result = []
    for d in desc:
        c = km.predict(d)
        result.append(
            np.array([np.sum(c==ci) for ci in xrange(k)],dtype=float)
        )
    
    return result
                           
def get_features(img,img_id):
    features = make_raw_features(img,img_id)
    img_id = features[0]
    haralick = features[1]
    color_hist = features[2]
    pftas = features[3]
    sift = bow(features[4],256,'SIFT')
    surf = bow(features[5],256,'SURF')
    lbp = features[6]
                           
    return [img_id,haralick,color_hist,pftas,sift,surf,lbp]
                           
def make_df(img_ids,img_haralick,img_chist,img_pftas,img_sift,img_surf,img_lbp):
    df = pd.DataFrame()
    df['ID'] = img_ids
    
    def make_columns_from_feature(feature,name,df):
        if name=='SIFT' or name=='SURF':
            for i in xrange(0,len(feature[0])):
                column=[]
                for j in xrange(0,len(feature)):
                    column.append(feature[j][i])
            
                column_name = str(name)+str(i+1)
                df[column_name] = column
        else:
            for i in xrange(0,len(feature)):
                column=[]
                column.append(feature[i])
                column_name = str(name)+str(i+1)
                df[column_name] = column
        return df
    
    df = make_columns_from_feature(img_haralick,'HARALICK',df)
    df = make_columns_from_feature(img_chist,'CHIST',df)
    df = make_columns_from_feature(img_pftas,'PFTAS',df)
    df = make_columns_from_feature(img_surf,'SURF',df)
    df = make_columns_from_feature(img_sift,'SIFT',df)
    df = make_columns_from_feature(img_lbp,'LBP',df)
    return df
                           
def transform_features(df):
    features = df[df.columns[1:]]
    LDA = joblib.load('./models/LDA.pkl')
    transformed_features = LDA.transform(features)
    return transformed_features

def etc_classify(features):                           
    ETc = joblib.load('./models/ETc.pkl')
    y_pred=ETc.predict(features)
    y_proba=ETc.predict_proba(features)
    return y_pred,y_proba

def classify(img,img_id):
    roi,image_detect = segmentation.segment_image(img)
    features = get_features(roi,img_id)
    df = make_df(features[0],features[1],features[2],features[3],features[4],features[5],features[6])
    transformed_features = transform_features(df)
    pred,proba = etc_classify(transformed_features)
    return image_detect, pred, proba
    
