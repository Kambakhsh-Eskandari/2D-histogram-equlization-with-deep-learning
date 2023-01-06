import numpy as np
import matplotlib.pyplot as plt
import cv2

##########################################################################################################################################################################


# hist
def imhist(arr):
    hist = np.zeros((256), dtype=np.uint32)
    for item in arr.copy().reshape(-1):
        hist[item] += 1 
    return hist

##########################################################################################################################################################################


# 2D hist
def im2dhist(image, w_neighboring = 6):
    V = image.copy()
    [h, w] = V.shape
    V_hist = imhist(V)
    
    X = (V_hist>0) * np.arange(1, 257)
    X = X[X>0]
    K = len(X)

    Hist2D = np.zeros((K, K))
    
    X_inv = np.zeros((X[-1], 1), dtype=np.uint8).reshape(-1)
    X_inv[X-1] = np.arange(1, K+1)
    
    for i in range(K):
        [xi, yi] = np.where(V==(X[i]-1))
        
        xi_n = np.zeros((xi.size, 2*w_neighboring+1), dtype=np.int16)
        yi_n = np.zeros((yi.size, 2*w_neighboring+1), dtype=np.int16)
        for ii in range(2*w_neighboring+1):
            xi_n[:, ii] = xi + (ii - w_neighboring)
            yi_n[:, ii] = yi + (ii - w_neighboring)
    
        
        xi_n = np.where(xi_n<h, xi_n, -1*np.ones_like(xi_n))
        yi_n = np.where(yi_n<w, yi_n, -1*np.ones_like(yi_n))
        
        
        for i_row in range(xi_n.shape[0]):
            xi_nr = xi_n[i_row, :].copy() 
            yi_nr = yi_n[i_row, :].copy() 
            xi_nr = xi_nr[np.where(xi_nr>=0)].copy()
            yi_nr = yi_nr[np.where(yi_nr>=0)].copy()
            
            neighboring_intens = V[xi_nr[0]:xi_nr[-1]+1, yi_nr[0]:yi_nr[-1]+1].copy().reshape(-1).astype(np.int16)
            
            for neighboring_inten in neighboring_intens:
                    Hist2D[i, X_inv[neighboring_inten]-1] += np.abs(neighboring_inten+1-X[i]) +1
    # Hist2D_normalized = Hist2D/np.sum(Hist2D)
    return Hist2D

##########################################################################################################################################################################



# 2D HE
def im2dhisteq(image, hist2d):
    V = image.copy()
    [h, w] = V.shape
    V_hist = imhist(V)
    H_in = hist2d/np.sum(hist2d)
    CDFx = np.cumsum(np.sum(H_in, axis=0)) # Kx1

    # normalizes CDFx
    CDFxn = (255*CDFx/CDFx[-1])

    PDFxn = np.zeros_like(CDFxn)
    PDFxn[0] = CDFxn[0]
    PDFxn[1:] = np.diff(CDFxn)

    X_transform = np.zeros((256))
    X_transform[np.where(V_hist > 0)] = PDFxn.copy()
    CDFxn_transform = np.cumsum(X_transform)


    # bins = np.array([i for i in range(0, 256)])
    bins = np.arange(256)
    # uses linear interpolation of cdf to find new pixel values
    image_equalized = np.floor(np.interp(V.flatten(), bins, CDFxn_transform).reshape(h, w)).astype(np.uint8)

    return image_equalized
##########################################################################################################################################################################



# def get_map(Y_1, Y_3, Y_5, hist2d_1, hist2d_3, hist2d_5):
#     Y_hist_1 = imhist(Y_1)
#     y_hist_1 = (Y_hist_1>0) * np.arange(1,257)
#     y_hist_1 = y_hist_1[y_hist_1>0]
#     L_1 = len(y_hist_1)

#     Y_hist_3 = imhist(Y_3)
#     y_hist_3 = (Y_hist_3>0) * np.arange(1,257)
#     y_hist_3 = y_hist_3[y_hist_3>0]
#     L_3 = len(y_hist_3)

#     Y_hist_5 = imhist(Y_5)
#     y_hist_5 = (Y_hist_5>0) * np.arange(1,257)
#     y_hist_5 = y_hist_5[y_hist_5>0]
#     L_5 = len(y_hist_5)

#     P_t_1 = np.full((L_1), 1/L_1)
#     P_t_3 = np.full((L_3), 1/L_3)
#     P_t_5 = np.full((L_5), 1/L_5)

#     P_x_1 = np.sum(hist2d_1/np.sum(hist2d_1), axis=0)
#     P_x_3 = np.sum(hist2d_3/np.sum(hist2d_3), axis=0)
#     P_x_5 = np.sum(hist2d_5/np.sum(hist2d_5), axis=0)
    
#     mapxy_1 = []
#     for m in range(len(P_x_1)):
#         P_t_copy_1 = P_t_1.copy()
#         P_t_copy_1 *= L_1
#         m_prime = np.min(abs(P_x_1[m] - P_t_copy_1))
#         mapxy_1.append(m_prime)
#     mapxy_1 = np.sort(mapxy_1)
    
#     mapxy_3 = []
#     for m in range(len(P_x_3)):
#         P_t_copy_3 = P_t_3.copy()
#         P_t_copy_3 *= L_3
#         m_prime = np.min(abs(P_x_3[m] - P_t_copy_3))
#         mapxy_3.append(m_prime)
#     mapxy_3 = np.sort(mapxy_3)

#     mapxy_5 = []
#     for m in range(len(P_x_5)):
#         P_t_copy_5 = P_t_5.copy()
#         P_t_copy_5 *= L_5
#         m_prime = np.min(abs(P_x_5[m] - P_t_copy_5))
#         mapxy_5.append((m_prime))
#     mapxy_5 = np.sort(mapxy_5) 
    
#     return mapxy_1, mapxy_3, mapxy_5, L_1, L_3, L_5


##########################################################################################################################################################################

# get map
def get_map_2dhe(Y_1, hist2d_1):
    
    Y_hist_1 = imhist(Y_1)
    y_hist_1 = (Y_hist_1>0) * np.arange(1,257)
    y_hist_1 = y_hist_1[y_hist_1>0]
    L_1 = len(y_hist_1)

    P_t_1 = np.full((L_1), 1/L_1)
    
    P_x_1 = np.sum(hist2d_1/np.sum(hist2d_1), axis=0)

    mapxy_1 = []
    
    for m in range(len(P_x_1)):
        P_t_copy_1 = P_t_1.copy()
        P_t_copy_1 *= L_1
        m_prime = np.min(abs(P_x_1[m] - P_t_copy_1))
        mapxy_1.append(m_prime)
    
    mapxy_1 = np.sort(mapxy_1)
    
    return mapxy_1*L_1
##########################################################################################################################################################################
def get_map(Y_1, hist1d_1):
    
    Y_hist_1 = imhist(Y_1)
    y_hist_1 = (Y_hist_1>0) * np.arange(1,257)
    y_hist_1 = y_hist_1[y_hist_1>0]
    L_1 = len(y_hist_1)

    P_t_1 = np.full((L_1), 1/L_1)
    
    P_x_1 = hist1d_1

    mapxy_1 = []
    
    for m in range(len(P_x_1)):
        P_t_copy_1 = P_t_1.copy()
        P_t_copy_1 *= L_1
        m_prime = np.min(abs(P_x_1[m] - P_t_copy_1))
        mapxy_1.append(m_prime)
    
    mapxy_1 = np.sort(mapxy_1)
    
    return mapxy_1*L_1
##########################################################################################################################################################################



# plot
def plot2dhist(Hist2D, title):
    # plot 2D-Histogram
    [K, _] = Hist2D.shape
    x = np.outer(np.arange(0, K), np.ones(K))
    y = x.copy().T 
    # ln-ing Hist2D makes its details more prominent.
    Hist2D_ln = Hist2D.copy()
    Hist2D_ln[np.where(Hist2D_ln<=0)] = 1e-15
    z = np.log(Hist2D_ln)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, z,cmap='viridis', edgecolor='none')
    ax.set_title(title)
    plt.show()
    
    
    
##########################################################################################################################################################################
####################################################################################################################################################################################################################################################################################################################################################

def calc_hist(image):
 
    hist = np.zeros(256)
    img_copy = image.copy() 
  
    img_list = list(img_copy.flatten())
  
    for p in img_list:
        hist[p] += 1
     
    return hist

##########################################################################################################################################################################

def post_process(cdf):
    cdf_copy = cdf.copy()
    for i in range(len(cdf_copy) - 1):
        if cdf_copy[i] > cdf_copy[i+1]:
            cdf_copy[i+1] = cdf_copy[i]
        if cdf_copy[i] > 1:
            cdf_copy[i] = 1
    if cdf_copy[255] > 1:
        cdf_copy[255] = 1
    return cdf_copy

##########################################################################################################################################################################

def calc_cdf(img):
    hist = calc_hist(img)
    n = np.sum(hist)
    hist_normal = hist / n
    cdf = np.cumsum(hist_normal)
    return cdf

##########################################################################################################################################################################


def AHE(img, const):
    hist = calc_hist(img)
    
    n = np.sum(hist)
    hist_normal = hist / n

    output_image=img.copy()
    
    
    hist_new = np.zeros(256,dtype=float)
    
    max = hist_normal.max()
    
    for p in range(0,256):

        if hist_normal[p] > const*max:
            hist_new[p] =  hist_normal[p] - const*max
            hist_normal[p] = const *max
    
    mean = np.mean(hist_new)

    hist_normal += mean

    cdf_normal = np.cumsum(hist_normal)
    transform_map = np.floor(255 * cdf_normal).astype(np.uint8)
    
    img_list_ = list(output_image.flatten())
    eq_img_list_ = [transform_map[p] for p in img_list_]

    output_image = np.reshape(np.asarray(eq_img_list_), output_image.shape)


    return output_image
    
    
##########################################################################################################################################################################


def HE(image):
    hist = calc_hist(image)
    hist_normal = hist / np.sum(hist)
    output_image = image.copy()
    cdf = np.cumsum(hist_normal)
    transform_map = np.floor(255 * cdf).astype(np.uint8)
    img_list = list(output_image.flatten())
    eq_image = [transform_map[p] for p in img_list]
    output_image = np.reshape(np.asarray(eq_image), output_image.shape)
    return output_image

##########################################################################################################################################################################

def apply_cdf(img, cdf):
    output_image = img.copy()
    transform_map = np.floor(255 * cdf).astype(np.uint8)
    img_list = list(output_image.flatten())
    eq_image = [transform_map[p] for p in img_list]
    output_image = np.reshape(np.asarray(eq_image), output_image.shape)
    return output_image
    
##########################################################################################################################################################################

def predict(image, model):
    image_copy = image.copy()
    
    image_copy = cv2.resize(image_copy, (120,120))
    image_copy = np.expand_dims(image_copy, axis=0)
    image_copy = np.expand_dims(image_copy, axis=3)
    pred = model.predict(image_copy)[0]
    pred = post_process(pred)
    return pred
##########################################################################################################################################################################
def predict_2dhe(image, model):
    
    image_copy = image.copy()
    image_copy = cv2.resize(image_copy, (120,120))
    image_copy = np.expand_dims(image_copy, axis=0)
    image_copy = np.expand_dims(image_copy, axis=3)
    pred = model.predict(image_copy)[0]
    pred /= 255
    pred = post_process(pred)
    return pred

##########################################################################################################################################################################

# comparison functions

def AMBE_n(X, Y):
    return 1/(1+ abs(np.mean(X) - np.mean(Y)))

##########################################################################################################################################################################

def DE_x(X):
    
    hist = calc_hist(X)
    hist_normal = hist / np.sum(hist)
    DE = 0
    for p in hist_normal:
        if p > 0:
            DE -= (p) * np.log(p)
    return DE
##########################################################################################################################################################################

def DE_n(X, Y):
    return 1 / (1 + (np.log(256) - DE_x(Y)) / (np.log(256) - DE_x(X)))

