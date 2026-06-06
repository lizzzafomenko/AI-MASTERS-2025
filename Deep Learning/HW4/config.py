common_config = {
    'data_dir': '/mnt/calc/lizzzafomenko/rcnn/license_plates',
    'img_width': 100,
    'img_height': 32,
    'map_to_seq_hidden': 64,
    'rnn_hidden': 256,
    'leaky_relu': False,
}

train_config = {
    'epochs': 50,
    'train_batch_size': 64,
    'eval_batch_size': 512,
    'lr': 1e-3,
    'cpu_workers': 4,
    'reload_checkpoint': None,
    'valid_max_iter': 100,
    'decode_method': 'greedy',
    'beam_size': 10,
    'checkpoints_dir': '/mnt/calc/lizzzafomenko/rcnn/checkpoints/',
    'load_ckpt': '/mnt/calc/lizzzafomenko/rcnn/crnn_synth90k.ckpt'
}
train_config.update(common_config)

evaluate_config = {
    'eval_batch_size': 512,
    'cpu_workers': 4,
    'reload_checkpoint': 'checkpoints/crnn_synth90k.pt',
    'decode_method': 'greedy',
    'beam_size': 10,
}

evaluate_config.update(common_config)