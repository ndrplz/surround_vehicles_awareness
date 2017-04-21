from model import SDPN
from keras.optimizers import Adam
from load_data import get_sample_batch
from utils import show_prediction
from keras.utils.data_utils import get_file


TH_WEIGHTS_PATH = 'http://imagelab.ing.unimore.it/files/pretrained_models/keras/SPDN_w.hdf5'


if __name__ == '__main__':

    # Get model
    model = SDPN(summary=True)

    # Download pre-trained weights
    pretrained_weights_path = get_file('SPDN_w.h5', TH_WEIGHTS_PATH, cache_subdir='models')

    # Load pre-trained weights
    model.load_weights(pretrained_weights_path)
    model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999), loss='mse')

    # Load sample batch
    X_coords, X_crops, X_images, X_images_original, Y_coords, Y_crops, Y_images, Y_dist, Y_yaw = get_sample_batch('data')

    # Perform prediction given (vehicle_coords, vehicle_crop) in dashboard camera view
    Y_pred = model.predict([X_coords, X_crops])

    # Display sample prediction
    for b in range(len(X_coords)):
        show_prediction(X_images_original[b], Y_images[b], X_coords[b], Y_coords[b], Y_pred[b])






