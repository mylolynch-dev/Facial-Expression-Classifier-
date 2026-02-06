from keras.utils import image_dataset_from_directory
from config import train_directory, test_directory, image_size, batch_size, validation_split

def _split_data(train_directory, test_directory, batch_size, validation_split):
    print('train dataset:')
    train_dataset, validation_dataset = image_dataset_from_directory(
        train_directory,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        validation_split=validation_split,
        subset="both",
        seed=47
    )
    print('test dataset:')
    test_dataset = image_dataset_from_directory(
        test_directory,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False
    )

    return train_dataset, validation_dataset, test_dataset

def get_datasets():
    train_dataset, validation_dataset, test_dataset = \
        _split_data(train_directory, test_directory, batch_size, validation_split)
    return train_dataset, validation_dataset, test_dataset

def get_transfer_datasets():
    print('transfer train dataset:')
    train_dataset = image_dataset_from_directory(
        'kaggle/transfer_train', 
        label_mode='binary',  # Binary classification for PNEUMONIA/NORMAL
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=47,
        validation_split=0.2,  # Split for validation
        subset="training"
    )

    print('transfer validation dataset:')
    validation_dataset = image_dataset_from_directory(
        'kaggle/transfer_train',  # Same directory as train with validation split
        label_mode='binary',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=47,
        validation_split=0.2,  # Split for validation
        subset="validation"
    )

    print('transfer test dataset:')
    test_dataset = image_dataset_from_directory(
        'kaggle/transfer_test',  
        label_mode='binary',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False
    )

    return train_dataset, validation_dataset, test_dataset