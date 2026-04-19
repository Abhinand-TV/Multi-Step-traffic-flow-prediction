class Config:
    DATA_PATH = "C:/Users/Dell/Downloads/traffic_flow/traffic_flow/METR-LA.h5"   # change if needed
    MODEL_PATH = "C:/Users/Dell/Downloads/traffic_flow/traffic_flow/model.pth"

    SEQ_LEN = 12
    PRED_LEN = 12

    BATCH_SIZE = 8
    EPOCHS = 25
    LR = 1e-3

    D_MODEL = 64
    N_HEADS = 4
    NUM_LAYERS = 2

    MAX_SAMPLES = 8000