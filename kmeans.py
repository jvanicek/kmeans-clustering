'''
Author: Joseph Vanciek
Date: 9/22/17
Class: ISTA 131

Description:

'''
import pandas as pd
from datetime import datetime
import numpy, math
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

'''
A series of functions that clean scale and cluster data to run the data through the k-means clustering algorithm
'''

def is_leap_year(year):
    '''
    Takes a year and returns true if it is a leap year and false if not.
    Parameters:
        year: year to determine leap year
    Returns: True or False
    '''
    return (year % 4 == 0 and year % 100 != 0) or year % 400 == 0
    
def euclidean_distance(fv1,fv2):
    '''
    Chacks for common features and Calcualtes the euclidean distance between two series objects.
    Parameters:
        fv1 = feature vector 1
        fv2 = feature vector 2
    returns; the euclidean distance or nan the two vectors do not have any common features. 
    '''
    features = set(fv1.index) & set(fv2.index)
    if len(features) > 0:
        dist = []
        for index in features:
            dist.append((fv1[index] - fv2[index])**2)
        result = sum(dist)   
        return math.sqrt(result)
        
    return numpy.nan
    
def make_frame():
    '''
    Makes a dataframe from the TIA data with datetime objects as the index
    Parameters: none
    Returns: df, the dataframe
    '''
    df = pd.read_csv('TIA_1987_2016.csv')
    dtstart = datetime(1987,1,1)
    dtend = datetime(2016,12,31)
    df.index = pd.date_range(dtstart,dtend)
    return df
    
def clean_dewpoint(mf):
    '''
    Cleans out the -999 values in the Dewpt colunm of the dataframe
    Parameters: mf the dataframe made by the previous function.
    Returns: None
    '''
    tenth = 0
    elev = 0
    for r in mf.index:
        if r.month == 3 and r.day == 10 and r.year != 2010:
            tenth+=(mf.loc[r,'Dewpt'])
        if r.month == 3 and r.day == 11 and r.year != 2010:
            elev+=(mf.loc[r,'Dewpt'])
    mf.loc[datetime(2010,3,10),'Dewpt'] = tenth/29
    mf.loc[datetime(2010,3,11),'Dewpt'] = elev/29
    
def day_of_year(dt):
    '''
    Takes a datetime object and returns the day of the year as a number between 1 and 365. 366, if date is 2/29 of a leap year.
    Parameters:
        dt: datetime object
    Returns: the day of year as a number between 1 and 365, 366 if date is 2/29 of a leap year.
    '''
    yday = dt.timetuple().tm_yday
    
    if not is_leap_year(dt.year) or yday <60:
        return yday
    if yday > 60:
        return yday-1
    else:
        return 366
        
def climatology(df):
    '''
    Creates a groupby object of the values from the date of year function to create a frame with the indecies of the day of the year,
    and the values are the the averages of of the values from the original frame.
    Parameters:
        df: dateframe
    Returns: the group object
    '''
    
    group = df.groupby(day_of_year).mean()
    return group.iloc[0:365]
    
def scale(df):
    '''
    Scales each of the featurs in a frame so that they run between 0 and 1
    Parameters:
        df: dataframe
    Returns:None
    '''
    MinMaxScaler(copy = False).fit_transform(df)
    
    
    
def get_initial_centroids(df,k):
    '''
    Creates a frame containing the starting values for the centroids.
    Parameters:
        df: dataframe
        k: The number of clusters we want to find.
    Returns: the new dataframe
    '''    
    indexlst = []
    for i in range(k):
        indexlst.append(list(df.iloc[i*int(len(df)/k)]))     
    new_df = pd.DataFrame( indexlst, columns=['Dewpt','AWS', 'Pcpn', 'MaxT', 'MinT'])
    return new_df
    
def classify(cf,fv):
    '''
    Finds the label of the cluster that is closest to the feature vector
    Parameters:
        cf: centroid dataframe
        fv: feature vector
    Returns: The closest label
    '''
    label = 0
    mindist = euclidean_distance(cf.iloc[0],fv)
    for i in range (1,len(cf)):
        newdist = euclidean_distance(cf.iloc[i],fv)
        if newdist < mindist:
            mindist = newdist
            label = i
    return label
    
def get_labels(df, cf):
    '''
    Creates a series that maps the indices of the fisrt argument(Days of the years) 
    to the labels of the clusters corresponding to that day
    Parameters:
        df: Dataframe (a scaled climatology frame)
        cf: centroid frame
    Returns: the series mapping the indices to the cluster labels
    '''
    label_lst = []
    for i in df.index:
        label_lst.append(classify(cf, df.loc[i]))
    return pd.Series(label_lst, df.index)
    
def update_centroids(df,cf,ls):
    '''
    Replaces every centroid in the centroid frame with the average of the data in its lcuster
    Parameters:
        df: DataFrame(a scaled climatology frame)
        cf: centroid frame
        ls: labels series
    Returns: None
    '''
    cf[:] = 0.0
    for i in ls.index:
        cf.loc[ls[i]] += df.loc[i]
    vc = ls.value_counts()
    for row in cf.index:
        cf.loc[row] /= vc[row]
        
def k_means(df,k):
    '''
    Calcualtes the actual centroids of the clusters as the centroids are moving, until the clusters and centroids are stable
    Parameters:
        df: DataFrame(a scaled climatology frame)
        k: k: The number of clusters
    Returns: centroid frame and leabels
    '''
    cf = get_initial_centroids(df,k)
    labels = get_labels(df,cf)
    while True:
        update_centroids(df,cf,labels)
        new_labels = get_labels(df,cf)
        if(labels == new_labels).all(): 
            return cf,labels
        labels = new_labels
    
    
def distortion(df, labels, centroids):
    '''
    measures how well a given clustering fits the data. It takes a data
    Parameters:
        df = DataFrame (a scaled climatology frame)
        labels = a labels Series
        centroids = centroid frame
    '''
    # return sum(euclidean_distance(df.loc[index],
    # centroids.loc[labels[index]])**2 for index in df.index)
    result = 0
    for index in df.index:
        label = labels[index]
        result += euclidean_distance(df.loc[index],centroids.loc[label])**2
    return result

def kmeans_list(df, max_k):
    '''
    takes a frame and a maximum k and returns a list of k-means dictionaries for k = 1 to max k. 
    Each k-means dictionary maps 'centroids' to the final centroids frame, 'labels' to the final labels Series, 
    'k' to k, and 'distortion' to the distortion for that k.
    '''
    list_of_kmeans = []
    for k in range(1, max_k + 1):
        centroids, labels = k_means(df, k)
        sum_distances = distortion(df, labels, centroids)
        list_of_kmeans.append({'centroids': centroids, 'labels':labels, 'k': k, 'distortion': sum_distances})
    return list_of_kmeans

def extract_distortion_dict(list_of_kmeans):
    '''
    takes a k-means list and returns a dictionary that maps values of k to the associated distortion.
    '''
    distortion_dict = {}
    for d in list_of_kmeans:
        distortion_dict[d['k']] = d['distortion']
    return distortion_dict
    
    
    
  
          
    

#==========================================================
def main():
    '''
    Calls the functions. Mainly used for testing
    '''
    raw = make_frame()
    clean_dewpoint(raw)
    climo = climatology(raw)
    scale(climo)
    list_of_kmeans = kmeans_list(climo, 10) # this will take a while
    distortion_dict = extract_distortion_dict(list_of_kmeans)
    distortion_series = pd.Series(distortion_dict)
    ax = distortion_series.plot()
    ax.set_ylabel('Distortion', size=24)
    ax.set_xlabel('k', size=24)

    k3 = list_of_kmeans[2] ['labels'] # list index starts at 0
    plt.figure()

    ax = k3[k3 == 0].plot(marker='o', linestyle=' ',color='navy', label='Cluster 1', markersize=5, yticks=[])
    k3[k3 == 1].plot(marker='o', linestyle=' ',color='#32CD32', label='Cluster 2', markersize=5, yticks=[])
    k3[k3 == 2].plot(marker='o', linestyle=' ',color='navy', label='Cluster 3', markersize=5, yticks=[])
    ax.set_title(r"$k$-means, $k$ = 3", fontsize=28)
    ax.set_xlabel('Day of Year', size=24)
    ax.legend(loc=2)
    ax.set_ylim(-0.5, 2.5)

    plt.show()

if __name__ == '__main__':
    main()
