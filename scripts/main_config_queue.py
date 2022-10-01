import fcntl
import os
import time
import json
import glob
import sys

import main


if __name__ == '__main__':
    configs_dir = './configs'

    mutex = open(os.path.join(configs_dir,'lock'),'w')
    success = False
    counter = 15

    while success == False and counter > 0:
        try:
            print('locking the dir')
            fcntl.lockf(mutex,fcntl.LOCK_EX)
        except:
            print('failed to lock')
            time.sleep(1)
            counter = counter - 1
            continue
        print('locked successfully')


        #try:
        config_list = glob.glob(os.path.join(configs_dir,'*.json'))
        config_path = config_list[0]
        os.rename(config_path, config_path+'.used')
        sys.stdout.flush()
        fcntl.lockf(mutex, fcntl.LOCK_UN)
        mutex.close()
        print('unlocked')

        with open(config_path+'.used','r') as f_conf:
            config_dict = json.load(f_conf)
            config_dict['name'] = config_dict['name']+'_'+os.path.basename(config_path)
            main.main(**config_dict)

        #except Exception as e:
        #    print(e)
        success = True

    print(f'success {success} counter {counter}')

    pass