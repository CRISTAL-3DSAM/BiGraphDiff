import tensorflow_gan as tfgan
from scipy.io import loadmat
import numpy as np
import math


def calculate_fvd(real_activations,
                  generated_activations):
    """Returns a list of ops that compute metrics as funcs of activations.
    Args:
      real_activations: <float32>[num_samples, embedding_size]
      generated_activations: <float32>[num_samples, embedding_size]
    Returns:
      A scalar that contains the requested FVD.
    """
    return tfgan.eval.frechet_classifier_distance_from_activations(
        real_activations, generated_activations)

def calculate_diversity(features):
    idx=np.arange(features.shape[0])
    seed = 2023
    np.random.seed(seed)
    np.random.shuffle(idx)
    if features.shape[0]%2==0.0:
        idx_1 = idx[:int(features.shape[0]/2)]
        idx_2 = idx[int(features.shape[0]/2):]
    else:
        idx_1 = idx[:math.floor(features.shape[0]/2)]
        idx_2 = idx[math.floor(features.shape[0]/2):features.shape[0]-1]
    features_1=features[idx_1]
    features_2=features[idx_2]
    nb_feat = features_1.shape[0]
    S=0
    for i in range(features_1.shape[0]):
        S=S+np.linalg.norm(features_1[i]-features_2[i])
    return S/nb_feat

def calculate_multimodality(features_full,classes):
    nb_class = 26
    nb_samples_pc = 100
    SS=0
    seed = 2021
    np.random.seed(seed)
    for c in range(nb_class):
        class_idx = np.where(classes==c)
        features = features_full[class_idx,:]
        idx=np.arange(features.shape[0])
        np.random.shuffle(idx)
        if features.shape[0]%2==0.0:
            idx_1 = idx[:int(features.shape[0]/2)]
            idx_2 = idx[int(features.shape[0]/2):]
        else:
            idx_1 = idx[:math.floor(features.shape[0]/2)]
            idx_2 = idx[math.floor(features.shape[0]/2):features.shape[0]-1]
        features_1=features[idx_1]
        features_2=features[idx_2]
        S=0
        for i in range(features_1.shape[0]):
            S=S+np.linalg.norm(features_1[i]-features_2[i])
        SS=SS+S

    return SS/(nb_class*nb_samples_pc)


GT = loadmat('features_NTU/features_GT_v2')#loadmat('features_NTU/features_GT_v2')
GT_class = GT['class']
GT = GT['features']

OURS = loadmat('features_NTU/features_OURS')
OURS_class = OURS['class']
OURS = OURS['features']

ACTOR = loadmat('features_NTU/features_ACTOR_v2')
ACTOR_class = ACTOR['class']
ACTOR = ACTOR['features']

MDIFF = loadmat('features_NTU/features_Mdiff_v2_1500')
MDIFF_class = MDIFF['class']
MDIFF = MDIFF['features']

FVD_ACTOR = calculate_fvd(GT,ACTOR)
FVD_OURS = calculate_fvd(GT,OURS)
FVD_MDIFF = calculate_fvd(GT,MDIFF)


MM_GT=calculate_multimodality(GT,GT_class)
MM_ACTOR=calculate_multimodality(ACTOR,ACTOR_class)
MM_OURS=calculate_multimodality(OURS,OURS_class)
MM_MDIFF=calculate_multimodality(MDIFF,MDIFF_class)

print('FVD of ACTOR:')
print(FVD_ACTOR.numpy())
print('\n')
print('\n')
print('FVD of MOTIONDIFFUSE:')
print(FVD_MDIFF.numpy())
print('\n')
print('FVD of IGDM:')
print(FVD_OURS.numpy())
print('\n')



print('Multimodality of ACTOR:')
print(100*np.abs(MM_ACTOR-MM_GT)/MM_GT)
print('\n')
print('Multimodality  of MOTIONDIFFUSE:')
print(100*np.abs(MM_MDIFF-MM_GT)/MM_GT)
print('\n')
print('Multimodality  of IGDM:')
print(100*np.abs(MM_OURS-MM_GT)/MM_GT)
print('\n')