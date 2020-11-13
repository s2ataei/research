from keras.layers import BatchNormalization as BN,Input,Conv2D,Activation,concatenate,add
from keras.models import Model

def build_DRL():
    
        inputs = Input(shape=(None,None,1))
        
        conv1 = Conv2D(64, (5,5), activation='relu', padding='same')(inputs)
        conv2 = Conv2D(64, (3, 3), padding='same',dilation_rate=(2,2))(conv1)
        conv2 = BN()(conv2)
        conv2 = Activation('relu')(conv2)
        
        conv3 = Conv2D(64, (3, 3), padding='same',dilation_rate=(3,3))(conv2)
        conv3 = BN()(conv3)
        conv3 = Activation('relu')(conv3)
        
        conv4 = Conv2D(64, (3, 3), padding='same',dilation_rate=(4,4))(conv3)
        conv4 = BN()(conv4)
        conv4 = Activation('relu')(conv4)
        
        conv5 = Conv2D(64, (3, 3), padding='same',dilation_rate=(3,3))(conv4)
        conv5 = BN()(conv5)
        conv5 = Activation('relu')(conv5)
        
        #Residual blocks via concatenate 
        conv6 = concatenate([conv5,conv2],axis=3)
        conv6 = Conv2D(64, (3, 3), padding='same',dilation_rate=(2,2))(conv6)
        conv6 = BN()(conv6)
        conv6 = Activation('relu')(conv6)
        
        conv7= concatenate([conv6,conv1],axis=3)
        conv8 = Conv2D(1, (3, 3), padding='same')(conv7)
    
        penultimate = add([conv8,inputs])
        
        output = Conv2D(1, (3, 3), padding='same')(penultimate)

        model = Model(inputs=[inputs], outputs=[output])
            
        model.summary()
        return model
    
