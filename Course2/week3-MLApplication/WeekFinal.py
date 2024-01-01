'''REGULARIZED NN'''

'''
# UNQ_C5
# GRADED CELL: model_r

tf.random.set_seed(1234)
model_r = Sequential(
    [
        ### START CODE HERE ### 
        Dense(120, activation='relu', name='layer1', kernel_regularizer=tf.keras.regularizers.l2(0.1)),
        Dense(40, activation='relu', name='layer2', kernel_regularizer=tf.keras.regularizers.l2(0.1)),
        Dense(6, activation='linear', name='layer3'),
        ### START CODE HERE ### 
    ], name= None
)
model_r.compile(
    ### START CODE HERE ### 
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=Adam(0.01),
    ### START CODE HERE ### 
)

model_r.fit(
    X_train, y_train,
    epochs=1000
)

model_r.summary()
'''