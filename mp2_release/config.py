
DEVICE = "cuda"
checkpoint_dir = "checkpoints/ec_840b300_lr_0_001"
batch_size = 32
num_epochs = 20
learning_rate = 0.001
train_data_path = "data/train.txt"
test_data_path = "data/test.txt"
val_data_path = "data/dev.txt"
hidden_data_path = "data/hidden.txt"
pose_set_file = "data/pos_set.txt"
tagset_file = "data/tagset.txt"
glove_name = "840B"
glove_dim = 300
pos_embedding_dim = 50
combine_type = "concatenate"

