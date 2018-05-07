def VGG19(img,out='MaxPooling2D'):
    '''
    Get the output of img via any layer of VGG19
    :param img: input layer
    :param out: conv output name
    :return: conv output 
    '''
    base_model = keras.applications.VGG19(input_tensor=img, include_top=False)
    return base_model.get_layer(out).output
