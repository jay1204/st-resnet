import argparse
from utils.data_helper import make_ucf_image_lst


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', action="store", dest="data", help='Specify the source files to generate rec file'
                                                                    'only [ucf-image, ucf-flow] are supported.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.data == 'ucf-image':
        make_ucf_image_lst()
    elif args.data == 'ucf-flow':
        pass
    else:
        raise NotImplementedError('This {} has not been supported.'.format(args.data))

    return

if __name__ == '__main__':


    main()