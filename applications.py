import keras

def VGG19(img,out='block5_pool'):
    '''
    Get the output of img via any layer of VGG19
    The structure of VGG19:https://pan.baidu.com/s/18Za5Cn_slJZuvim2yZGvAA
    :param img: input layer
    :param out: conv output name
    :return: conv output
    '''
    base_model = keras.applications.VGG19(input_tensor=img, include_top=False)
    return base_model.get_layer(out).output

def VGG16(img,out='block5_pool'):
    '''
    Get the output of img via any layer of VGG16
    The structure of VGG16:https://pan.baidu.com/s/1k36U8qRtt3VH5v4P4XSKLg
    :param img: input layer
    :param out: conv output name
    :return: conv output
    '''
    base_model = keras.applications.VGG16(input_tensor=img, include_top=False)
    return base_model.get_layer(out).output
