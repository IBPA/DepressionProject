# -*- coding: utf-8 -*-
"""The default configuration for MSAP. User are free to modify this file to
customize the configuration.

Attributes:
    COLUMNS_CATEGORICAL (list): List of categorical columns.
    PARAMS_OD (dict): Parameters for outlier detection methods.
    PARAMS_MVI (dict): Parameters for missing value imputation methods.

Authors:
    Fangzhou Li - fzli@ucdavis.edu

"""
# Experiment parameters.
# COLUMNS_CATEGORICAL = ['fake_bi_cat_feature', 'fake_tri_cat_feature']
# COLUMNS_CATEGORICAL = ["Gender"]
# 12to18
# COLUMNS_CATEGORICAL = ['mom_married', 'mom_nomarry', 'severe_dep_m', 'momlostjob_yn', 'pills_yn_m', 'ptnrlostjob_yn', 'Child_sex', 'mom_anx', 'mom_dep', 'partner_anx', 'partner_anx_m', 'ptnr_dep', 'ptnr_dep_m', 'ptnrsamejob', 'pastmo_momdep', 'pastyr_pdep', 'A4d4:Childcurrentlyhasaspecialteacher', 'DV:Anymoodsymptomsyes/no', "B1n1:Mother'snaturalmotherhaseverhaddepressiony/n", 'C3d:Wife/partnerhashaddepression,inlast2years', 'mom_married_6', 'SMFQ_dep']
# 144m
# COLUMNS_CATEGORICAL = ['mom_div', 'mom_married', 'mom_wid', "NATmotherknowntohavedepressionetc.(aboutmother'smother)", 'severe_dep_m', 'momlostjob_yn', 'momsep_yn', 'pills_anx_yn_m', 'pills_yn_m', 'ptnrdeath_yn', 'mom_currpartner', 'Mother_ethnic', 'Child_sex', 'mom_anx', 'mom_dep', 'partner_anx', 'partner_anx_m', 'ptnrnewjob_yn', 'ptnrsamejob', 'mom_partnered', 'pastmo_momdep', 'C7a:Mothercurrentlymarriedorlivingwithpartner', 'DV:Childhaseverbeenconsideredforstatementing', 'DV:Childhaseverhaddevelopmentaldelay', 'DV:Childhaseverhademotional&behaviouraldifficulties', 'DV:Anymoodsymptomsyes/no', 'DV:DAWBADSM-IVclinicaldiagnosis-AnxietydisorderNOS', 'DV:DAWBADSM-IVclinicaldiagnosis-Anydepressivedisorder', 'DV:DAWBADSM-IVclinicaldiagnosis-DisruptivebehaviourdisorderNOS', 'B13b:Childhasafavouritefriendoftheothersex', 'B14:Childisateasewithchildrenofownage', 'C3d:Wife/partnerhashaddepression,inlast2years', 'C3e:Wife/partnerhashadanxietyornerves,inlast2years', 'SMFQ_dep_ft']
# 162m
# COLUMNS_CATEGORICAL = ['mom_livewpartner', 'mom_sep', 'mom_wid', 'momdivorce_yn', 'pills_yn_m', 'ptnrdeath_yn', 'Child_sex', 'mom_anx', 'mom_dep', 'partner_anx_m', 'ptnr_dep_m', 'pastmo_momdep', 'C7a:Mothercurrentlymarriedorlivingwithpartner', 'A4d4:Childcurrentlyhasaspecialteacher', 'DV:Childhaseverhademotional&behaviouraldifficulties', 'DV:Anymoodsymptomsyes/no', 'DV:DAWBADSM-IVclinicaldiagnosis-Specificphobia', 'C1d:Samepartner/husbandasmotherhadwhenstudychildhad6thbirthday', 'SMFQ_dep_ft']
# 192m
# COLUMNS_CATEGORICAL = ['mom_sep', 'mom_wid', 'severe_dep_m', 'momlostjob_yn', 'momsep_yn', 'pills_yn_m', 'ptnraway_yn', 'ptnrdeath_yn', 'Child_sex', 'mom_anx', 'mom_dep', 'momnewjob_yn', 'notwanted', 'partner_anx', 'partner_anx_m', 'ptnr_dep', 'ptnr_dep_m', 'p_wmom', 'partner_employ', 'pastmo_momdep', 'pastyr_pdep', 'A4d4:Childcurrentlyhasaspecialteacher', 'DV:ChildhaseverhadotherSENproblem', 'DV:DAWBADSM-IVclinicaldiagnosis-DepressivedisorderNOS', 'DV:DAWBADSM-IVclinicaldiagnosis-DisruptivebehaviourdisorderNOS', 'reprodpb', 'B13b:Childhasafavouritefriendoftheothersex', 'C3d:Wife/partnerhashaddepression,inlast2years', 'SMFQ_dep_ft']
# 204m
# COLUMNS_CATEGORICAL = ['mom_livewpartner', 'mom_married', 'mom_nomarry', 'mom_wid', "NATmotherknowntohavedepressionetc.(aboutmother'smother)", 'severe_dep_m', 'momlostjob_yn', 'mommarry_yn', 'momsep_yn', 'pills_yn_m', 'ptnrlostjob_yn', 'Child_ethnic', 'mom_currpartner', 'Child_sex', 'mom_anx', 'mom_dep', 'notwanted', 'partner_anx_m', 'ptnr_dep', 'ptnr_dep_m', 'liveswith_dad', 'mom_partnered', 'partner_employ', 'pastmo_momdep', 'C7a:Mothercurrentlymarriedorlivingwithpartner', 'DV:Anymoodsymptomsyes/no', 'DV:DAWBADSM-IVclinicaldiagnosis-AnyADHDdisorder', 'DV:DAWBADSM-IVclinicaldiagnosis-Anyoppositional-conductdisorder', 'DV:DAWBADSM-IVclinicaldiagnosis-Oppositionaldefiantdisorder', 'SMFQ_dep_ft']
# 216m
# COLUMNS_CATEGORICAL = ['mom_div', 'mom_livewpartner', 'mom_married', 'mom_nomarry', 'mom_sep', 'severe_dep_m', 'momdivorce_yn', 'momsep_yn', 'pills_yn_m', 'ptnrdeath_yn', 'ptnrlostjob_yn', 'mom_currpartner', 'Child_sex', 'hospital_adm', 'mom_anx', 'mom_dep', 'partner_anx_m', 'ptnr_dep_m', 'liveswith_dad', 'mom_partnered', 'pastmo_momdep', 'DV:Anymoodsymptomsyes/no', 'DV:DAWBADSM-IVclinicaldiagnosis-Anyanxietydisorder', 'DV:DAWBADSM-IVclinicaldiagnosis-Pervasivedevelopmentdisorder', 'reprodpb', 'B13b:Childhasafavouritefriendoftheothersex', 'C3d:Wife/partnerhashaddepression,inlast2years', 'C3e:Wife/partnerhashadanxietyornerves,inlast2years']
# 12to18 rfe results
COLUMNS_CATEGORICAL = ['Child_sex', 'mom_dep', 'partner_anx_m']

# Hyperparameters for the outlier detection methods.
PARAMS_OD = {
    'iforest': {
        'n_estimators': 100,
        'contamination': 'auto',
        'random_state': 42,
    },
    'none': {},
}

# Hyperparameters for the missing value imputation methods.
PARAMS_MVI = {
    'locf': {},
    'nocb': {},
    'simple': {},
}

# Hyperparameters for the grid search.
# for basicmotions
# PARAMS_GRID = {
#     'tsf': {
#         'n_estimators': [10, 50],
#         'criterion': ['entropy'],
#         'max_depth': [5, 10],
#         'min_samples_split': [5],
#         'min_samples_leaf': [1],
#         'random_state': [42],
#     },
#     'rnn': {
#         'num_layers': [1, 2],
#         'hidden_size': [100, 200],
#         'dropout': [0.1, 0.2],
#         'batch_size': [2, 3],
#         'max_epochs': [1, 10],
#         'lr': [0.001, 0.005],
#         'optimizer': ['AdamW'],
#         'criterion': ['CrossEntropyLoss'],
#         'random_state': [42],
#     },
#     'lstm': {
#         'num_layers': [1, 2],
#         'hidden_size': [100, 200],
#         'dropout': [0.1, 0.2],
#         'batch_size': [2, 3],
#         'max_epochs': [1, 10],
#         'lr': [0.001, 0.005],
#         'optimizer': ['AdamW'],
#         'criterion': ['CrossEntropyLoss'],
#         'random_state': [42],
#     }
# }
# for physionet
# PARAMS_GRID = {
#     'tsf': {
#         'n_estimators': [10, 50],
#         'criterion': ['entropy'],
#         'max_depth': [5, 10],
#         'min_samples_split': [5],
#         'min_samples_leaf': [1],
#         'random_state': [42],
#     },
#     'rnn': {
#         'num_layers': [1, 2],
#         'hidden_size': [20, 40, 80],
#         'batch_size': [16, 64],
#         'max_epochs': [10],
#         'lr': [0.001, 0.005],
#         'optimizer': ['AdamW'],
#         'criterion': ['CrossEntropyLoss'],
#         'random_state': [42],
#     },
#     'lstm': {
#         'num_layers': [1, 2],
#         'hidden_size': [20, 40, 80],
#         'batch_size': [16, 64],
#         'max_epochs': [10],
#         'lr': [0.001, 0.005],
#         'optimizer': ['AdamW'],
#         'criterion': ['CrossEntropyLoss'],
#         'random_state': [42],
#     }
# }
# dep grid search
PARAMS_GRID = {
    'tsf': {
        'n_estimators': [10, 50],
        'criterion': ['entropy'],
        'max_depth': [5, 10],
        'min_samples_split': [5],
        'min_samples_leaf': [1],
        'random_state': [42],
    },
    'rnn': {
        'num_layers': [2, 4],
        'hidden_size': [100, 120, 140],
        'dropout': [0.25, 0.5],
        'batch_size': [16, 64],
        'max_epochs': [50, 75, 100],
        'lr': [0.000001, 0.00001],
        'optimizer': ['AdamW'],
        'criterion': ['CrossEntropyLoss'],
        'random_state': [42],
    },
    'lstm': {
        'num_layers': [2, 4],
        'hidden_size': [100, 120, 140],
        'dropout': [0.25, 0.5],
        'batch_size': [16, 64],
        'max_epochs': [50, 75, 100],
        'lr': [0.000001, 0.00001],
        'optimizer': ['AdamW'],
        'criterion': ['CrossEntropyLoss'],
        'random_state': [42],
    }
}
# PARAMS_GRID = {
#     'tsf': {
#         'n_estimators': [30, 40, 50, 60, 70],
#         'criterion': ['entropy'],
#         'max_depth': [3],
#         'min_samples_split': [20],
#         'min_samples_leaf': [10],
#         'random_state': [42],
#     },
# }
