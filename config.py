import torchvision.transforms as T
IMAGE_WIDTH = 864
IMAGE_HEIGHT = 480
LEARNING_RATE = 0.002
MOMENTUM = 0.9

EPOCHS = 30

transforms = T.Compose([
    T.Resize(224, 0),
    T.ToTensor(),
])

TEST_BATCH = 16
VALIDATION_BATCH = 4
TEST_BATCH = 4
