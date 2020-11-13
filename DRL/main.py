import h5py
import numpy as np

from DRL import build_DRL

from keras.callbacks import ModelCheckpoint 
from keras.optimizers import Adam


PATH_TO_LABELS = ''
PATH_TO_INPUTS = ''

# Load h5 data 
f = h5py.File(PATH_TO_LABELS)
for k, v in f.items():
    y_train = np.array(v)
                              
f = h5py.File(PATH_TO_INPUTS)
for k, v in f.items():            
    X_train = np.array(v)  
    
# Reshape data to Batch x Height x Width x Channel        
X_train  = np.moveaxis(X_train, -1, 0)
y_train = np.moveaxis(y_train, -1, 0)
X_train = np.expand_dims(X_train, axis=3)
y_train = np.expand_dims(y_train, axis=3)
    
# Build model
model = build_DRL()

# Save weights after every epoch
model_checkpoint_callback = ModelCheckpoint(
    filepath=r'K:\FC-AIDE\weights\diffs_MSE\Epoch{epoch}loss{val_loss}.h5',
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=False)

# Initialize optimizer 
ADAM=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# Compile Model
model.compile(optimizer=ADAM,loss='mse')

# Train model
model.fit(x=X_train, y=y_train, batch_size=32, epochs=20, verbose=1, shuffle=True, validation_split=0.1, callbacks=[model_checkpoint_callback])
