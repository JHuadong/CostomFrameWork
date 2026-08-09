def get_conv_outsize(input_size, kernel_size, stride, pad):
    return (input_size + pad * 2 - kernel_size) // stride + 1

if __name__ == '__main__':
    H, W = 4, 4  # input_shape
    KH, KW = 3, 3  # kernel_shape
    SH, SW = 1, 1  # stride（Vertical stride length, horizontal stride length）
    PH, PW = 1, 1  # padding（Vertical padding, horizontal padding）
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)
    print(OH, OW)