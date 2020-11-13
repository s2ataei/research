from keras.applications import VGG16
from keras import losses
from keras.models import Model
from keras import backend as K


def perceptual_loss(y_true, y_pred):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=(256,256,3))
    selectedLayers = ['block1_conv2','block2_conv2','block3_conv3','block4_conv3']
    selectedOutputs = [vgg.get_layer(i).output for i in selectedLayers]
    
    loss_model = Model(inputs=vgg.input, outputs=selectedOutputs)
    loss_model.trainable = False
    
    mse2 = losses.mean_squared_error(y_true,y_pred)
    mse = K.variable(value=0)
    
    for i in range(0,3):
        mse = mse+ K.mean(K.square(loss_model(y_true)[i] - loss_model(y_pred)[i]))
        
    total_loss = (0.1*mse) + mse2 

    return total_loss