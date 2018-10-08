from __future__ import print_function

import os
import numpy as np
from keras.layers import Dense, concatenate,Cropping1D,GlobalAveragePooling1D
from keras.layers import Bidirectional, Input, BatchNormalization,CuDNNLSTM
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger,EarlyStopping
from Tools.Data_Processing_Tools_h5d import read_file_paths,GenSequence_AVG,read_test_data,GenSequence_Test,GenSequence_Test_long
from keras.utils import multi_gpu_model
import argparse
from Cavg_computations import Compute_Cavg


### Paramets Define
num_class =10
time_step = 98
input_dim =39
label_size = 1

r_dir_path = '/media/eleceng/E/Sarith/Data_OLR18/FDLP/'
train_file_name = './Lists/Features_Seg_Train_list.txt'
val_file_name = './Lists/Features_Seg_Val_list.txt'


### Arg_Parser
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=400, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--save_dir', default='FDLP_BLSTM')
parser.add_argument('--ncpu', default=24, type=int)
parser.add_argument('--ngpu', default=2, type=int)
parser.add_argument('--stage', default=0, type=int)
args = parser.parse_args()
print(args)

if not os.path.exists('./'+ args.save_dir):
    os.makedirs(args.save_dir)

os.system("cp %s %s" % (__file__, './'+ args.save_dir+'/'+args.save_dir+'.py'))

if args.ngpu<2:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"


### Model Define
def network_initialization(num_class, time_step,input_dim):
    print('Build model...')
    main_input = Input(shape=(time_step, input_dim), name='main_input')

    x_1 = Cropping1D((0, 4))(main_input)
    x_2 = Cropping1D((1, 3))(main_input)
    x_3 = Cropping1D((2, 2))(main_input)
    x_4 = Cropping1D((3, 1))(main_input)
    x_5 = Cropping1D((4, 0))(main_input)

    x = concatenate([x_1, x_2, x_3, x_4, x_5])

    x = BatchNormalization(name='fc1')(x)
    x1 = Bidirectional(CuDNNLSTM(1024, return_sequences=True), name='fc2')(x)

    x2 = BatchNormalization(name='fc3')(x1)
    x2 = GlobalAveragePooling1D()(x2)

    main_output = Dense(num_class, activation='softmax', name='main_output')(x2)

    model = Model(inputs=[main_input], outputs=[main_output])

    single_model=model

    if args.stage < 1:
        model.summary()

    if args.ngpu>1:
        model = multi_gpu_model(model, gpus=args.ngpu)

    return model, single_model


### Training

#Constructing Generator
save_path = './' + args.save_dir + '/weights.best.model'
batch_size=args.batch_size*args.ngpu

### Training
if args.stage < 1:

    dir_path =  r_dir_path + 'train.h5'
    train_file = read_file_paths(train_file_name)
    train_generator = GenSequence_AVG(train_file, dir_path, batch_size=batch_size,num_class=num_class, time_step=time_step,
                                input_dim=input_dim, label_size=label_size,shuffle=True)
    train_steps_per_epoch=int(np.floor(len(train_file)/batch_size))

    val_file = read_file_paths(val_file_name)
    val_generator = GenSequence_AVG(val_file, dir_path, batch_size=batch_size , num_class=num_class, time_step=time_step,
                              input_dim=input_dim, label_size=label_size,shuffle=False)
    val_steps_per_epoch=int(np.floor(len(val_file)/batch_size))

    #Building Model
    model, single_model = network_initialization(num_class, time_step, input_dim)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    #Callbacks

    checkpoint = ModelCheckpoint(save_path, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True,
                                 mode='auto', period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=0, min_lr=0.0001)
    csv_logger = CSVLogger('./'+ args.save_dir+'/training.txt', separator=',', append=False)
    early_st = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
    callbacks_list = [checkpoint, reduce_lr, csv_logger, early_st]

    #Training Model
    model.fit_generator(train_generator,steps_per_epoch=train_steps_per_epoch, epochs=args.epochs, validation_data=val_generator,
                        validation_steps=val_steps_per_epoch, callbacks=callbacks_list ,max_queue_size=10,
                       workers=args.ncpu, use_multiprocessing=True)
    single_model.save('./' + args.save_dir + '/weights.best.single.model')

    print('Training completed')

if not os.path.exists('./scores'):
    os.makedirs(args.save_dir)


### Testing
if args.stage <2:
    args.stage=1
    ### Testing

    f= open('./' + args.save_dir + '/Results_ped_gen.txt','w')
    f.write('Results Cavg')
    save_score = './scores/'

    time_step=[98,98,298,298]
    time_dues=['dev_1s','test_1s','dev_3s','test_3s']

    for due in range (0,4):
        list_test = read_file_paths('./Lists/' +time_dues[due] + '_list.txt')
        dir_path = r_dir_path + time_dues[due] +'.h5'
        model, single_model = network_initialization(num_class,time_step[due],input_dim)
        model.load_weights(save_path)


        test_steps = int(np.floor(len(list_test) / batch_size))
        generator_preds = test_steps*batch_size
        class_t = np.zeros((len(list_test), num_class))

        print('Generator Prediction...')
        test_generator= GenSequence_Test(list_test[:generator_preds], dir_path, batch_size, time_step[due], input_dim)
        class_t[:generator_preds] = model.predict_generator(test_generator,steps=test_steps ,max_queue_size=10,
                       workers=args.ncpu, use_multiprocessing=True,verbose=1)

        print('Model Prediction...')
        testX = read_test_data(dir_path, list_test[generator_preds:], time_step[due], input_dim)
        class_t[generator_preds:] = model.predict(testX)

        np.savetxt(save_score + 'Scroes_'+ time_dues[due] + '_'+ args.save_dir, class_t, fmt='%.6f', delimiter=',',newline='\n')
        print('Testing completed ' + time_dues[due])

        f.write('\n' + time_dues[due] + ' > ')
        f.write(Compute_Cavg(save_score + 'Scroes_' + time_dues[due] + '_' + args.save_dir, time_dues[due]))

    f.close()

### Testing
if args.stage < 3:
    args.stage = 2
    args.ngpu =1
    ### Testing
    save_path = './' + args.save_dir + '/weights.best.single.model'
    f = open('./' + args.save_dir + '/Results_all_pred_gen_test.txt', 'w')
    f.write('Results Cavg all')
    save_score = './scores/'


    time_dues = ['dev_all','test_all']

    for due in range(0, 2):

        list_test = read_file_paths('./Lists/' + time_dues[due] + '_list.txt')
        dir_path = r_dir_path + time_dues[due] + '.h5'
        model, single_model = network_initialization(num_class, None , input_dim)
        model.load_weights(save_path)

        batch_size_long = 1
        test_steps = int(np.floor(len(list_test) / batch_size_long))
        generator_preds = test_steps * batch_size_long
        class_t = np.zeros((len(list_test), num_class))

        print('Generator Prediction...')
        test_generator = GenSequence_Test_long(list_test[:generator_preds], dir_path, batch_size_long, input_dim)
        class_t[:generator_preds] = model.predict_generator(test_generator, steps=test_steps, max_queue_size=10,
                                                            workers=args.ncpu, use_multiprocessing=True, verbose=1)


        np.savetxt(save_score + 'Scroes_' + time_dues[due] + '_' + args.save_dir, class_t, fmt='%.6f', delimiter=',',
                   newline='\n')
        print('Testing completed ' + time_dues[due])


        f.write('\n' + time_dues[due] + ' > ')
        f.write(
            Compute_Cavg(save_score + 'Scroes_' + time_dues[due] + '_' + args.save_dir, time_dues[due]))

    f.close()

