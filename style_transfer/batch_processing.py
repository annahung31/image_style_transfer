import os
import argparse
import glob
from train import network_train
from test import network_test

def build_parser():
    parser = argparse.ArgumentParser()

    # cpu, gpu mode selection
    parser.add_argument('--gpu-no', type=int,
                    help='cpu : -1, gpu : 0 ~ n ', default=0)

    ### arguments for network training
    parser.add_argument('--train', action='store_true',
                    help='Train flag', default=False)

    parser.add_argument('--max-iter', type=int,
                    help='Train iterations', default=40000)  #40000

    parser.add_argument('--batch-size', type=int,
                    help='Batch size', default=1)

    parser.add_argument('--lr', type=float,
                    help='Learning rate to optimize network', default=1e-3)

    parser.add_argument('--check-iter', type=int,
                    help='Number of iteration to check training logs', default=100)

    parser.add_argument('--imsize', type=int,
                    help='Size for resize image during training', default=None)

    parser.add_argument('--cropsize', type=int,
                    help='Size for crop image durning training', default=None)

    parser.add_argument('--cencrop', action='store_true',
                    help='Flag for crop the center rigion of the image, default: randomly crop', default=False)

    parser.add_argument('--layers', type=int, nargs='+',
                    help='Layer indices to extract features', default=[1, 6, 11, 20])

    parser.add_argument('--feature-weight', type=float,
                    help='Feautre loss weight', default=0.1)

    parser.add_argument('--tv-weight', type=float,
                    help='Total valiation loss weight', default=1.0)

    parser.add_argument('--content-dir', type=str,
                    help='Content data path to train the network')

    parser.add_argument('--save-path', type=str,
                    help='Save path', default='./trained_models/')

    parser.add_argument('--check-point', type=str, default= './pretrain_model/check_point.pth',
                    help="Trained model load path")

    parser.add_argument('--content', type=str, default= None, nargs='+',
                    help="Test content image path")

    parser.add_argument('--fig-name', type=str,
                    help="file name of stylized image")

    parser.add_argument('--style', type=str,default= None, nargs='+',
                    help="Test style image path")
    
    parser.add_argument('--mask', type=str, nargs='+',
                    help="Mask image for masked stylization", default=None)

    parser.add_argument('--style-strength', type=float,
                    help='Content vs style interpolation value: 1(style), 0(content)', default=0.2)

    parser.add_argument('--interpolation-weights', type=float, nargs='+',
                    help='Multi-style interpolation weights', default=None)

    parser.add_argument('--patch-size', type=int,
                    help='Size of patch for swap normalized content and style features',  default=3)

    parser.add_argument('--patch-stride', type=int,
                    help='Size of patch stride for swap normalized content and style features',  default=1)

    parser.add_argument('--img_root', type=str, default='/volume/annahung-project/share/20200130', help='folder of content/style/output images')

    return parser

if __name__ == '__main__':
    parser = build_parser()
    args= parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_no)


    assert args.train == False
    ROOT = args.img_root
    content_folder = sorted(glob.glob(os.path.join(ROOT,'/content/*.jpg')))
    
    style_folder = sorted(glob.glob(os.path.join(ROOT,'/style/*.jpg')))
    output_folder = os.path.join(ROOT,'/output/')
    
    for args.content in content_folder:
        c_file_name = args.content.split('/')[-1].split('.')[0]
        
        for style_img in style_folder:
            args.style = [style_img]
            s_file_name = style_img.split('/')[-1].split('.')[0]
            print('Turn <<{}>> into <<{}>> style'.format(c_file_name, s_file_name))
            args.fig_name = output_folder + c_file_name + '_to_' + s_file_name +'.jpg'
            try:
                network_test(args)
            except:
                print('fig too big, not processed')
