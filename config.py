N_CLASS = {
    'angio'    :2,
    'cad'      :2,
    'whole'    :2,
    'full'     :0,
}

DATA_TYPES = {
    'angio'    : ['angio'],
    'cad'      : ['angio', 'cad'],
    'whole'    : ['angio', 'cad', 'whole'],
    'full'     : ['full']
}

TRAINER = {
    'finetune':'SupervisedTrainer',
    'linear':'SupervisedTrainer',
    'randominit':'SupervisedTrainer',
    'BYOL':'BYOL',
    'SimCLR':'SimCLR',
    'CMSC':'CMSC',
    'OURS':'OURS'
}