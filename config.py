N_CLASS = {
    'angio'    :2,
    'non_angio':2,
    'whole'    :2,
    'full'     :0,
}

DATA_TYPES = {
    'angio'    : ['angio'],
    'non_angio': ['angio', 'non_angio'],
    'whole'    : ['angio', 'non_angio', 'whole'],
    'full'     : None
}

TRAINER = {
    'finetune':'SupervisedTrainer',
    'linear':'SupervisedTrainer',
    'randominit':'SupervisedTrainer',
    'BYOL':'BYOL',
    'SimCLR':'SimCLR',
}