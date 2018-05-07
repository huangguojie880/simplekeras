from keras.utils import generic_utils

def simple_progbar(steps):
    '''
    :param steps: Step inside an epoch
    :return:-
    '''
    progbar = generic_utils.Progbar(steps)
    def progbar_update(status):
        '''
        :param status: a list,The first parameter is the current step;the second parameter is dict, corresponding to each value
        :return:-
        '''
        key_value = []
        for (d, x) in dict.items(status[1]):
            key_value.append((d,x))
        progbar.update(status[0],key_value)
    return progbar_update
