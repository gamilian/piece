'''
This file save all of hyperparameters
'''

# NOTE: directory must end at '/', because we use this path to call external c++ program
WorkSpacePath = {
    "training_dataset_root": "your path",
    "testing_dataset_root": "your path",
    "example_testing_root": "../Examples",
    "example_evaluate_root": "../Evaluate",
    "checkpoint_dir": "/work/csl/code/piece/checkpoints/JigsawCNN_checkpoint"
}

# for CNN
NNHyperparameters = {
    "width": 160,       # image width
    "height": 160,      # image height
    "depth" : 3,        # action candidates + original image
    "batch_size": 64,
    "weight_decay": 1e-4,
    "learning_rate": 1e-4,
    "total_training_step": 30000,
    "learner_num": 5
}